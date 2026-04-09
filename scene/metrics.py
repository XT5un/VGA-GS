from typing import Dict, Literal

import torch
from skimage.metrics import structural_similarity as ssim_sk
from torch import Tensor

from submodules.lpipsPyTorch.modules.lpips import LPIPS

from .losses import ssim

__all__ = ["MultiMetrics"]


class MultiMetrics:
    def __init__(
        self, device: torch.device, lpips_net_type: Literal["vgg", "alex"] = "vgg"
    ) -> None:
        # self._lpips_fn = LPIPS(net="alex").to(device).eval()
        self._lpips_fn = LPIPS(net_type=lpips_net_type, version="0.1").to(device)

    def compute_psnr(
        self,
        image_pred: Tensor,
        image_gt: Tensor,
        image_mask: Tensor | None,
    ) -> float:
        if image_mask is not None:
            image_mask = image_mask.float()[..., None]
            image_pred = image_pred * image_mask + (1 - image_mask)
            image_gt = image_gt * image_mask + (1 - image_mask)

        mse = (image_pred - image_gt) ** 2
        mse = torch.mean(mse)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def compute_ssim(
        self, image_pred: Tensor, image_gt: Tensor, image_mask: Tensor | None
    ) -> float:
        image_pred = image_pred[None, ...].permute(0, 3, 1, 2)
        image_gt = image_gt[None, ...].permute(0, 3, 1, 2)
        if image_mask is not None:
            image_mask = image_mask.float()
            image_mask = image_mask[None, None, ...]
        return ssim(image_pred, image_gt, mask=image_mask).item()

    def compute_ssim_sk(
        self,
        image_pred: Tensor,
        image_gt: Tensor,
        image_mask: Tensor | None,
    ) -> float:
        if image_mask is not None:
            image_mask = image_mask.float()[..., None]
            image_pred = image_pred * image_mask + (1 - image_mask)
            image_gt = image_gt * image_mask + (1 - image_mask)
        image_pred = image_pred.cpu().numpy()
        image_gt = image_gt.cpu().numpy()
        return ssim_sk(image_pred, image_gt, channel_axis=2, data_range=1.0).item()

    def compute_lpips(
        self,
        image_pred: Tensor,
        image_gt: Tensor,
        image_mask: Tensor | None,
    ) -> float:
        if image_mask is not None:
            image_mask = image_mask.float()[..., None]
            image_pred = image_pred * image_mask + (1 - image_mask)
            image_gt = image_gt * image_mask + (1 - image_mask)
        image_pred = image_pred[None, ...].permute(0, 3, 1, 2)
        image_gt = image_gt[None, ...].permute(0, 3, 1, 2)
        return self._lpips_fn(image_pred, image_gt).item()

    def compute_depth_rmse(
        self, depth_pred: Tensor, depth_gt: Tensor, depth_mask: Tensor | None
    ) -> float:
        if depth_mask is not None:
            depth_pred = depth_pred.clone()
            depth_gt = depth_gt.clone()
            depth_pred = depth_pred[depth_mask]
            depth_gt = depth_gt[depth_mask]
        return torch.sqrt(torch.mean((depth_pred - depth_gt) ** 2)).item()

    def __call__(
        self,
        image_pred: Tensor,
        image_gt: Tensor,
        depth_pred: Tensor,
        depth_gt: Tensor | None,
        image_mask: Tensor | None = None,
        depth_mask: Tensor | None = None,
    ) -> Dict[str, float]:
        psnr_score = self.compute_psnr(image_pred, image_gt, image_mask)
        ssim_score = self.compute_ssim(image_pred, image_gt, image_mask)
        lpips_score = self.compute_lpips(image_pred, image_gt, image_mask)
        if depth_gt is None:
            rmse_score = 0.0
        else:
            rmse_score = self.compute_depth_rmse(depth_pred, depth_gt, depth_mask)
        return {
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpips": lpips_score,
            "rmse": rmse_score,
        }
