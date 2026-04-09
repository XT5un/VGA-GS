from math import exp
from typing import Tuple

import torch
import torch.nn.functional as F
from kornia.filters import median_blur

# from pytorch_msssim import ssim
from torch import Tensor
from torch.autograd import Variable

__all__ = [
    "compute_l1_loss",
    "compute_l2_loss",
    "compute_ssim_loss",
    "compute_group_depth_rank_loss",
    "compute_normal_loss",
    "compute_local_smoothness_loss",
    "compute_local_normal_loss",
]


def compute_local_smoothness_loss(
    image: Tensor,
    weight: Tensor | None = None,
    patch_size: int = 3,
) -> Tensor:
    if image.ndim == 2:
        median_filtered = median_blur(image[None, None, ...], patch_size)
        median_filtered = median_filtered[0, 0].detach()
        loss = torch.abs(median_filtered - image)
    elif image.ndim == 3:
        median_filtered = median_blur(image.permute(2, 0, 1)[None, ...], patch_size)
        median_filtered = median_filtered[0].permute(1, 2, 0).detach()
        loss = torch.mean(torch.abs(median_filtered - image), dim=-1)
    if weight is not None:
        loss *= weight
    loss = torch.mean(loss)
    return loss


def compute_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    loss = torch.abs(pred - target)
    loss = torch.mean(loss)
    return loss


def compute_l2_loss(pred: Tensor, target: Tensor) -> Tensor:
    loss = (pred - target) ** 2
    loss = torch.mean(loss)
    return loss


def compute_ssim_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred = pred[None, ...].permute(0, 3, 1, 2)
    target = target[None, ...].permute(0, 3, 1, 2)
    return 1 - ssim(pred, target)


def compute_local_normal_loss(normal: Tensor) -> Tensor:
    center_normal = normal[1:-1, 1:-1]  # (H-2, W-2, 3)
    top_normal = normal[:-2, 1:-1]
    bottom_normal = normal[2:, 1:-1]
    left_normal = normal[1:-1, :-2]
    right_normal = normal[1:-1, 2:]

    stacked_normal = torch.stack(
        [
            top_normal,
            bottom_normal,
            left_normal,
            right_normal,
        ],
        dim=-2,
    )  # (H-2, W-2, 4, 3)
    center_normal = center_normal[:, :, None, :]
    diff = torch.abs(torch.sum(center_normal * stacked_normal, dim=-1))
    diff = torch.min(diff, dim=-1)[0]
    loss = torch.mean(diff)
    return loss


def compute_group_depth_rank_loss(
    depth_pred: Tensor, depth_target: Tensor, group_level: int = 32, mask: Tensor = None
) -> Tensor:

    def _sort_by_group(
        pred: Tensor, target: Tensor, group_level: int
    ) -> Tuple[Tensor, Tensor]:
        pred_ = pred.reshape(-1)
        target_ = target.reshape(-1)

        total_elements = len(pred_) // group_level * group_level
        ids = torch.randperm(len(pred_), device=pred.device)[:total_elements]
        pred_ = pred_[ids]
        target_ = target_[ids]

        sorted_ids = torch.argsort(target_).reshape(group_level, -1)
        pred_ = pred_[sorted_ids]
        target_ = target_[sorted_ids]
        # shuffle the each group
        shuffle_ids = torch.randperm(pred_.shape[1], device=pred.device)
        pred_ = pred_[:, shuffle_ids].transpose(0, 1)  # (num_per_group, group_level)
        target_ = target_[:, shuffle_ids].transpose(0, 1)

        return pred_, target_

    if mask is not None:
        depth_pred = depth_pred[mask]
        depth_target = depth_target[mask]

    pred, _ = _sort_by_group(
        depth_pred, depth_target, group_level
    )  # (num_per_group, group_level)

    ids_i = torch.arange(group_level, device=pred.device)[:, None]
    ids_j = torch.arange(group_level, device=pred.device)[None, :]
    pred_i = pred[:, ids_i]
    pred_j = pred[:, ids_j]

    sign = torch.sign(ids_i - ids_j).to(pred.dtype)
    diff_pred = pred_i - pred_j
    rank_error_matrix = torch.clamp(-sign[None, :] * diff_pred, min=0)
    loss = torch.mean(rank_error_matrix)

    return loss


def compute_normal_loss(
    normal_pred: Tensor,
    normal_target: Tensor,
    mask: Tensor | None = None,
    ignore_edge: bool = False,
) -> Tensor:
    if ignore_edge:
        assert normal_pred.ndim == normal_target.ndim == 3
        normal_pred = normal_pred[1:-1, 1:-1]
        normal_target = normal_target[1:-1, 1:-1]
        if mask is not None:
            mask = mask[1:-1, 1:-1]

    if mask is not None:
        normal_pred = normal_pred[mask]
        normal_target = normal_target[mask]

    normal_pred = normal_pred / torch.norm(normal_pred, dim=-1, keepdim=True)
    normal_target = normal_target / torch.norm(normal_target, dim=-1, keepdim=True)

    normal_diff = 1 - torch.sum(normal_pred * normal_target, dim=-1)
    normal_diff = torch.nan_to_num(normal_diff, nan=0.0)

    loss = torch.mean(normal_diff)
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
