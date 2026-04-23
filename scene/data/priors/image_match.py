import os
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import torch
from numpy import ndarray
from torch import Tensor

_GIM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../submodules/gim")
)
if not os.path.exists(_GIM_DIR):
    raise ImportError("GIM not found")
elif _GIM_DIR not in sys.path:
    sys.path.append(_GIM_DIR)
_ROOT_DIR = os.path.abspath(os.path.join(_GIM_DIR, "../.."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from submodules.gim.dkm.models.model_zoo.DKMv3 import DKMv3


class BaseMatcher(ABC):

    MIN_NUM_MATCHES = 4
    RANSAC_MAX_ITER = 10000
    RANSAC_CONFIDENCE = 0.999999
    RANSAC_REPROJ_THRESHOLD = 1.0
    RANSAC_METHOD = cv2.USAC_MAGSAC

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._model = self._get_model().to(self._device).eval()

    @abstractmethod
    def predict(
        self,
        image0: Tensor | ndarray,
        image1: Tensor | ndarray,
        filter_by_fundamental: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplemented("Subclasses must implement this method")

    @abstractmethod
    def _get_model(self) -> torch.nn.Module:
        raise NotImplemented("Subclasses must implement this method")

    def _filter_by_fundamental(self, kpts0: Tensor, kpts1: Tensor) -> Tensor:
        assert kpts0.shape[0] == kpts1.shape[0]

        kpts0 = kpts0.cpu().numpy()
        kpts1 = kpts1.cpu().numpy()

        if kpts0.shape[0] < self.MIN_NUM_MATCHES:
            return None

        _, mask = cv2.findFundamentalMat(
            kpts0,
            kpts1,
            method=self.RANSAC_METHOD,
            ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD,
            confidence=self.RANSAC_CONFIDENCE,
            maxIters=self.RANSAC_MAX_ITER,
        )
        mask = mask.ravel() > 0
        mask = torch.from_numpy(mask).to(self._device)

        return mask


class DKMModel(BaseMatcher):

    SPARSE_MATCHES = 2_000

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)

    def _get_model(self) -> torch.nn.Module:
        model = DKMv3(weights=None, h=672, w=896)
        ckpt_path = os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/gim_dkm_100h.ckpt"
        )
        assert os.path.exists(ckpt_path), f"{ckpt_path} not found"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict[k.replace("model.", "", 1)] = state_dict.pop(k)
            if "encoder.net.fc" in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model

    def _preprocess(self, image: Tensor | ndarray) -> Tensor:
        if isinstance(image, ndarray):
            image = torch.from_numpy(image)
        image = image.to(self._device)
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @torch.no_grad()
    def predict(
        self,
        image0: Tensor | ndarray,
        image1: Tensor | ndarray,
        filter_by_fundamental: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        image0 = self._preprocess(image0)
        image1 = self._preprocess(image1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense_matches, dense_certainty = self._model.match(image0, image1)
            sparse_matches, mconf = self._model.sample(
                dense_matches, dense_certainty, self.SPARSE_MATCHES
            )

        height0, width0 = image0.shape[-2:]
        height1, width1 = image1.shape[-2:]
        kpts0 = sparse_matches[:, :2]
        kpts1 = sparse_matches[:, 2:]
        kpts0 = torch.stack(
            (width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2),
            dim=-1,
        )
        kpts1 = torch.stack(
            (width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2),
            dim=-1,
        )

        kpts0[..., 0] = torch.clamp(kpts0[..., 0], 0, width0 - 1)
        kpts0[..., 1] = torch.clamp(kpts0[..., 1], 0, height0 - 1)
        kpts1[..., 0] = torch.clamp(kpts1[..., 0], 0, width1 - 1)
        kpts1[..., 1] = torch.clamp(kpts1[..., 1], 0, height1 - 1)

        if filter_by_fundamental:
            mask = self._filter_by_fundamental(kpts0, kpts1)
            kpts0 = kpts0[mask]
            kpts1 = kpts1[mask]

        return kpts0, kpts1


# if __name__ == "__main__":
#     from utils import write_match

#     device = torch.device("cuda")
#     model = DKMModel(device)
#     image_dir = "/hdd24T/sunxt/dataset/Scannet_DDP/scene0710_00/train/rgb"
#     image0_path = os.path.join(image_dir, "1415.jpg")
#     image1_path = os.path.join(image_dir, "1461.jpg")
#     image0 = cv2.cvtColor(cv2.imread(image0_path), cv2.COLOR_BGR2RGB)
#     image1 = cv2.cvtColor(cv2.imread(image1_path), cv2.COLOR_BGR2RGB)
#     image0 = torch.tensor(image0, dtype=torch.float32, device=device) / 255.0
#     image1 = torch.tensor(image1, dtype=torch.float32, device=device) / 255.0

#     kpts0, kpts1 = model.predict(image0, image1)
#     write_match(
#         "match.png",
#         image0.cpu().numpy(),
#         image1.cpu().numpy(),
#         kpts0.cpu().numpy(),
#         kpts1.cpu().numpy(),
#         max_num=500,
#     )
