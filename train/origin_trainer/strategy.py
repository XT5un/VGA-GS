from typing import Any, Dict

import torch
from torch.optim import Optimizer

from scene.gaussians import Gaussians3D

try:
    from base_trainer import BaseRefinementStrategy
except:
    from ..base_trainer.strategy import BaseRefinementStrategy

from .configs import OriginConfigs


class OriginRefinementStrategy(BaseRefinementStrategy):
    def __init__(
        self,
        config: OriginConfigs,
        step: int,
        optimizer: Optimizer,
        gaussians: Gaussians3D,
        scene_radius: float,
    ):
        self._optimizer = optimizer
        self._gaussians = gaussians
        self._step = step

        self._refine_from_step = config.refine_from_step
        self._refine_until_step = config.refine_until_step

        self._densify_interval = config.densify_interval
        self._densify_grad_threshold = config.densify_grad_threshold
        self._densify_scale_threshold = config.percent_dense * scene_radius

        self._prune_opacity_threshold = 0.1
        self._prune_scale_threshold = 0.1 * scene_radius
        self._prune_screen_threshold = 20

        self._reset_opacity_interval = config.reset_opacity_interval
        self._reset_opacity_threshold = config.reset_opacity_threshold

        self._accum_grads = None
        self._max_radii = None
        self._denom = None

    def __update_state(self, end_points: Dict[str, Any]) -> None:
        assert "means2d" in end_points, "means2d not found in end_points"
        assert "radii" in end_points, "radii not found in end_points"

        grads = end_points["means2d"].grad
        radii = end_points["radii"]

        if self._accum_grads is None:
            self._accum_grads = torch.zeros(
                len(self._gaussians),
                dtype=torch.float32,
                device=self._gaussians.device,
            )
        if self._max_radii is None:
            self._max_radii = torch.zeros(
                len(self._gaussians),
                dtype=torch.int32,
                device=self._gaussians.device,
            )
        if self._denom is None:
            self._denom = torch.zeros(
                len(self._gaussians),
                dtype=torch.int32,
                device=self._gaussians.device,
            )
        vis_index = torch.where(radii > 0)[0]

        self._accum_grads.index_add_(0, vis_index, grads[vis_index].norm(dim=-1))
        self._denom.index_add_(0, vis_index, torch.ones_like(radii[vis_index]))
        self._max_radii[vis_index] = torch.maximum(
            self._max_radii[vis_index], radii[vis_index]
        )

    def __clone(self, avg_grads: torch.Tensor) -> None:
        # 如果一个高斯梯度很大但是scale很小，那么这个高斯就是需要被clone的
        high_grad_mask = avg_grads >= self._densify_grad_threshold
        scales = torch.exp(self._gaussians.get_scales).max(-1).values
        small_scale_mask = scales <= self._densify_scale_threshold
        clone_mask = high_grad_mask & small_scale_mask
        self._gaussians.clone(self._optimizer, clone_mask)
        # 重置states
        self._accum_grads = torch.zeros(
            len(self._gaussians),
            dtype=torch.float32,
            device=self._gaussians.device,
        )
        self._max_radii = torch.zeros(
            len(self._gaussians),
            dtype=torch.int32,
            device=self._gaussians.device,
        )
        self._denom = torch.zeros(
            len(self._gaussians),
            dtype=torch.int32,
            device=self._gaussians.device,
        )

    def __split(self, avg_grads: torch.Tensor) -> None:
        # 如果一个高斯梯度很大但是scale很大，那么这个高斯就是需要被split的
        high_grad_mask = avg_grads >= self._densify_grad_threshold
        scales = torch.exp(self._gaussians.get_scales).max(-1).values
        large_scale_mask = scales > self._densify_scale_threshold
        split_mask = high_grad_mask & large_scale_mask
        self._gaussians.split(self._optimizer, split_mask)
        # 重置states
        self._accum_grads = torch.zeros(
            len(self._gaussians),
            dtype=torch.float32,
            device=self._gaussians.device,
        )
        self._max_radii = torch.zeros(
            len(self._gaussians),
            dtype=torch.int32,
            device=self._gaussians.device,
        )
        self._denom = torch.zeros(
            len(self._gaussians),
            dtype=torch.int32,
            device=self._gaussians.device,
        )

    def __prune(self) -> None:
        opacities = torch.sigmoid(self._gaussians.get_opacities).squeeze(-1)
        low_opacity_mask = opacities < self._prune_opacity_threshold
        scales = torch.exp(self._gaussians.get_scales).max(-1).values
        large_scale_mask = scales > self._prune_scale_threshold
        if self._step > self._reset_opacity_interval:
            large_screen_mask = self._max_radii > self._prune_screen_threshold
        else:
            large_screen_mask = torch.zeros_like(low_opacity_mask)

        prune_mask = low_opacity_mask | large_scale_mask | large_screen_mask
        self._gaussians.prune_parameters(self._optimizer, prune_mask)
        # 重置states
        self._accum_grads = torch.zeros(
            len(self._gaussians),
            dtype=torch.float32,
            device=self._gaussians.device,
        )
        self._max_radii = torch.zeros(
            len(self._gaussians),
            dtype=torch.int32,
            device=self._gaussians.device,
        )
        self._denom = torch.zeros(
            len(self._gaussians),
            dtype=torch.int32,
            device=self._gaussians.device,
        )

    def __densify_and_prune(self) -> None:
        avg_grads = self._accum_grads / self._denom
        avg_grads[avg_grads.isnan()] = 0.0

        self.__clone(avg_grads)
        # print(f"After clone: {len(self._gaussians)}")
        # clone后高斯数量变化，对grads进行padding
        num_gaussians = len(self._gaussians)
        padded_avg_grads = torch.zeros(num_gaussians, device=avg_grads.device)
        padded_avg_grads[: len(avg_grads)] = avg_grads
        self.__split(padded_avg_grads)
        # print(f"After split: {len(self._gaussians)}")
        self.__prune()
        # print(f"After prune: {len(self._gaussians)}")
        torch.cuda.empty_cache()

    def _reset_opacity(self) -> None:
        opacities = torch.sigmoid(self._gaussians.get_opacities).clone()
        new_opacities = torch.clamp_max(opacities, self._reset_opacity_threshold)
        new_opacities = torch.logit(new_opacities)
        self._gaussians.reset_parameters(self._optimizer, {"opacities": new_opacities})

    @torch.no_grad()
    def step(self, end_points: Dict[str, Any]) -> None:
        if self._step < self._refine_until_step:
            self.__update_state(end_points)

            if (
                self._step > self._refine_from_step
                and self._step % self._densify_interval == 0
            ):
                self.__densify_and_prune()

            if (
                self._step % self._reset_opacity_interval == 0
                and self._step > 0
                or self._step == self._refine_from_step
            ):
                self._reset_opacity()

        self._step += 1
