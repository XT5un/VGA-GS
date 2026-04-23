import os
import sys
from abc import ABC, abstractmethod
from typing import List, Literal, Tuple

import cv2
import depth_pro
import numpy as np
import torch
import torchvision.transforms as T
from numpy import ndarray
from torch import Tensor

# ! DepthAnythingV2 is not a pip package, import from local
_DEPTH_ANYTHING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../submodules/Depth_Anything_V2")
)
if not os.path.exists(_DEPTH_ANYTHING_DIR):
    raise ImportError("Depth-Anything not found")
elif _DEPTH_ANYTHING_DIR not in sys.path:
    sys.path.append(_DEPTH_ANYTHING_DIR)
_ROOT_DIR = os.path.abspath(os.path.join(_DEPTH_ANYTHING_DIR, "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from submodules.Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import (
    DepthAnythingV2,
)


class BaseDepthEstimator(ABC):
    def __init__(self, device: torch.device, **kwargs) -> None:
        self._device = device

    @abstractmethod
    def _get_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def predict(self, image: Tensor | ndarray) -> Tensor:
        pass


class DepthProModel(BaseDepthEstimator):

    def __init__(self, device: torch.device, **kwargs) -> None:
        self._device = device
        model = self._get_model()
        self._model = model.to(self._device).eval()

    def _get_model(self) -> torch.nn.Module:
        config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/depth_pro.pt"
        )
        model, _ = depth_pro.create_model_and_transforms(config)
        return model

    @torch.no_grad()
    def predict(self, image: Tensor | ndarray) -> Tensor:
        if isinstance(image, ndarray):
            image = torch.from_numpy(image)

        image = image.to(self._device).permute(2, 0, 1)[None, ...]
        image = (image - 0.5) / 0.5
        prediction = self._model.infer(image)
        depth = prediction["depth"]
        return depth


class DepthAnythingModel(BaseDepthEstimator):

    def __init__(
        self,
        device: torch.device,
        scene_type: Literal["indoor", "outdoor"] = "indoor",
        **kwargs,
    ) -> None:
        self._device = device
        self._scene_type = scene_type
        self._model = self._get_model().to(self._device).eval()

    def _get_model(self) -> torch.nn.Module:
        # https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        encoder = "vitl"  # or 'vits', 'vitb'
        dataset = "hypersim" if self._scene_type == "indoor" else "vkitti"
        max_depth = 20 if self._scene_type == "indoor" else 80

        model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
        ckpt_path = os.path.expanduser(
            f"~/.cache/torch/hub/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        return model

    @torch.no_grad()
    def predict(self, image: Tensor | ndarray) -> Tensor:
        if torch.is_tensor(image):
            image = image.cpu().numpy()
            image = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        depth = self._model.infer_image(image)
        depth = torch.from_numpy(depth).to(self._device)
        return depth


class Metric3DModel(BaseDepthEstimator):

    _INPUT_SIZE = (616, 1064)  # for vit

    def __init__(self, device: torch.device, **kwargs) -> None:
        self._device = device
        self._model = self._get_model().to(self._device).eval()
        self._input_trans = self._get_input_transform()

    def _get_model(self) -> torch.nn.Module:
        repo_or_dir = os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")
        model = "metric3d_vit_giant2"
        source = "local"
        if not os.path.exists(repo_or_dir):
            repo_or_dir = "yvanyin/metric3d"
            source = "github"
        model = torch.hub.load(repo_or_dir, model, source=source, pretrain=True)
        return model

    def _get_input_transform(self) -> T.Compose:
        mean = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]
        std = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]
        return T.Compose([T.Normalize(mean=mean, std=std)])

    def _preprocess(
        self, image: Tensor | ndarray
    ) -> Tuple[Tensor, Tuple[int, int, int, int], Tuple[int, int]]:
        if isinstance(image, ndarray):
            image = torch.from_numpy(image)
        image = image.to(self._device)
        image = image.permute(2, 0, 1)
        image = self._input_trans(image)
        raw_h, raw_w = image.shape[-2:]
        scale = min(self._INPUT_SIZE[0] / raw_h, self._INPUT_SIZE[1] / raw_w)
        image = torch.nn.functional.interpolate(
            image[None, ...],
            size=(int(raw_h * scale), int(raw_w * scale)),
            mode="bilinear",
        )
        # padding
        h, w = image.shape[-2:]
        top = (self._INPUT_SIZE[0] - h) // 2
        bottom = self._INPUT_SIZE[0] - h - top
        left = (self._INPUT_SIZE[1] - w) // 2
        right = self._INPUT_SIZE[1] - w - left
        padding = (left, right, top, bottom)
        image = torch.nn.functional.pad(image, padding, mode="constant", value=0)

        return image, padding, (raw_h, raw_w)

    def _postprocess(
        self,
        data: Tensor,
        padding: Tuple[int, int, int, int],
        raw_size: Tuple[int, int],
    ) -> Tensor:
        h, w = data.shape[-2:]
        h_start = padding[2]
        h_end = h - padding[3]
        w_start = padding[0]
        w_end = w - padding[1]
        data = data[:, :, h_start:h_end, w_start:w_end]
        data = torch.nn.functional.interpolate(data, raw_size, mode="nearest")
        data = data.squeeze(dim=0).permute(1, 2, 0)
        return data

    @torch.no_grad()
    def predict(
        self, image: Tensor | ndarray, return_normal: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        image, padding, raw_size = self._preprocess(image)
        pred_depth, confidence, output_dict = self._model.inference({"input": image})
        pred_depth = self._postprocess(pred_depth, padding, raw_size)
        pred_depth = pred_depth.squeeze(dim=2)

        if not return_normal:
            return pred_depth

        pred_normal = output_dict["prediction_normal"][:, :3, :, :]
        pred_normal = self._postprocess(pred_normal, padding, raw_size)
        return pred_depth, pred_normal


# if __name__ == "__main__":
#     import cv2
#     import numpy as np

#     device = torch.device("cuda")
#     model = Metric3DModel(device)
#     image_dir = "/hdd24T/sunxt/dataset/Scannet_DDP/scene0710_00/train/rgb"

#     focal_length = 500.0

#     for fname in os.listdir(image_dir):
#         image = cv2.imread(os.path.join(image_dir, fname))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = image.astype(np.float32) / 255.0

#         pred_depth = model.predict(image)

#         H, W = pred_depth.shape
#         pred_depth = pred_depth.cpu().numpy()

#         # depth to point cloud
#         x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
#         x = (x - W / 2) * pred_depth / focal_length
#         y = (y - H / 2) * pred_depth / focal_length
#         z = pred_depth
#         pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
#         np.savetxt(f"{fname}.xyz", pts, fmt="%.6f")
