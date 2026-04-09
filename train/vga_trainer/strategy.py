from typing import Any, Dict, List

import numpy as np
import torch
from base_trainer import BaseRefinementStrategy
from configs import VGAGSConfigs
from kornia.filters import canny
from pytorch3d.ops import knn_points, sample_farthest_points
from simple_knn._C import distCUDA2
from torch.optim import Optimizer

from scene.data import GroundTruthDataset
from scene.gaussians import RGB2SH, Gaussians3D, GaussiansRayV2, generate_random_quats
from scene.priors_manager import PriorsManager
from scene.renderer import RendererFunction, RenderResult


def sample_alone_edge(
    image: torch.Tensor,
    depth: torch.Tensor,
    ps: int = 0,
    color_low_th: float = 0.1,
    color_high_th: float = 0.2,
    depth_low_th: float = 0.1,
    depth_high_th: float = 0.2,
) -> torch.Tensor:
    color_edge = canny(
        image[None, ...].permute(0, 3, 1, 2),
        color_low_th,
        color_high_th,
    )[1]
    color_edge = color_edge[0, 0, ...].bool()

    # normalize depth to [0, 1]
    depth_mask = depth > 0
    d_min = torch.min(depth[depth_mask])
    d_max = torch.max(depth[depth_mask])
    depth = (depth - d_min) / (d_max - d_min)
    depth_edge = canny(
        depth[None, None, ...],
        depth_low_th,
        depth_high_th,
    )[1]
    depth_edge = depth_edge[0, 0, ...].bool()

    if ps <= 0:
        return color_edge | depth_edge

    uniform_mask = torch.zeros_like(depth_edge)
    uniform_mask[ps // 2 :: ps, ps // 2 :: ps] = True
    return color_edge | depth_edge | uniform_mask


def _align_depth_via_lstsq(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        src_ = source[mask]
        tgt_ = target[mask]
    else:
        src_ = source
        tgt_ = target

    src_ = src_.view(-1, 1)
    tgt_ = tgt_.view(-1, 1)
    X = torch.cat([src_, torch.ones_like(src_)], dim=1)
    y = tgt_
    sln = torch.linalg.lstsq(X, y).solution
    A, B = sln[0, 0], sln[1, 0]

    aligned = A * source + B

    return aligned


class VGAGSRefinementStrategy(BaseRefinementStrategy):
    def __init__(
        self,
        config: VGAGSConfigs,
        step: int,
        optimizer: Optimizer,
        gaussians: GaussiansRayV2 | Gaussians3D,
        train_set: GroundTruthDataset,
        priors: PriorsManager,
        renderer: RendererFunction,
        bg_color: torch.Tensor,
    ):
        self._optimizer = optimizer
        self._step = step
        self._gaussians = gaussians
        self._train_set = train_set
        self._priors = priors
        self._renderer = renderer
        self._bg_color = bg_color

        self._prune_opacity_from = config.prune_opacity_from
        self._prune_opacity_until = config.prune_opacity_until
        self._prune_opacity_interval = config.prune_opacity_interval
        self._prune_opacity_threshold = config.prune_opacity_threshold

        self._reset_scale_until = config.reset_scale_until
        self._reset_scale_interval = config.reset_scale_interval

        self._densify_from_view_at = config.densify_from_view_at
        self._sample_important_pixels = config.sample_important_pixels
        self._patch_size_for_sample = config.patch_size_for_sample
        self._max_num_splatter_per_view = config.max_num_splatter_per_view
        self._downsample_voxel_res = config.downsample_voxel_res

        self._hard_color_th = 0.1
        self._max_added_pixels_per_view = 2000
        self._farthest_sample_ratio = 0.1

    @property
    def _degree_to_use(self) -> int:
        """Degree of SH to use when this step."""
        return min(self._step // 1000, self._gaussians.sh_degree)

    def __prune_low_opacity(self) -> None:

        opacities = torch.sigmoid(self._gaussians.get_opacities.detach()).squeeze()
        prune_mask = opacities < self._prune_opacity_threshold
        if prune_mask.any():
            self._gaussians.prune_parameters(self._optimizer, prune_mask)

    @torch.no_grad()
    def __densify_from_view(self) -> None:

        # TODO: 这里要修改成兼容gaussians_3d的代码

        added_cam_ids: List[torch.Tensor] = []
        added_uv: List[torch.Tensor] = []
        added_quats: List[torch.Tensor] = []
        added_depths: List[torch.Tensor] = []
        added_points: List[torch.Tensor] = []
        added_features_dc: List[torch.Tensor] = []
        added_features_rest: List[torch.Tensor] = []

        all_old_quats = self._gaussians.get_quats.clone()
        all_old_points = self._gaussians.get_means.clone()

        for cam_id in range(len(self._train_set)):
            camera = self._train_set[cam_id]
            render_rlt: RenderResult = self._renderer(
                self._gaussians.get_render_parameters(self._degree_to_use),
                camera.model,
                camera.w2c,
                self._bg_color,
                need_extra_infos=False,
                require_coord=False,
            )

            color_error: torch.Tensor = torch.abs(render_rlt.image - camera.image)
            color_error = torch.max(color_error, dim=-1).values
            error_mask = color_error > self._hard_color_th

            error_v, error_u = torch.where(error_mask)
            if len(error_v) < 50:
                continue
            elif len(error_v) > self._max_added_pixels_per_view:
                sample_num = int(len(error_v) * self._farthest_sample_ratio)
                error_uv = torch.stack([error_u, error_v], dim=1).float() + 0.5
                sample_uv = sample_farthest_points(error_uv[None, ...], K=sample_num)[0]
                sample_uv = sample_uv[0, ...]
            else:
                sample_uv = torch.stack([error_u, error_v], dim=1).float() + 0.5

            if isinstance(self._gaussians, GaussiansRayV2):
                old_depth, old_uv, cam_mask = self._gaussians.get_sub_sparse_depths(
                    cam_id, return_mask=True
                )
                old_quats = all_old_quats[cam_mask]
                # old_feat_dc = all_old_features_dc[cam_mask]
                # old_feat_rest = all_old_features_rest[cam_mask]
                # old_points = all_old_points[cam_mask]

                nn_ret = knn_points(sample_uv[None, ...], old_uv[None, ...], K=1)
                new_quats = old_quats[nn_ret.idx[0, :, 0]]
                new_depth = old_depth[nn_ret.idx[0, :, 0]]
                new_points = camera.depth2points(new_depth, sample_uv, to_world=True)

                sample_colors = camera.image[
                    sample_uv[:, 1].int(), sample_uv[:, 0].int()
                ]
                new_feat_dc = RGB2SH(sample_colors[:, None, :])
                new_feat_rest = torch.zeros(
                    len(new_depth),
                    self._gaussians.sh_rest_dim,
                    3,
                    device=sample_colors.device,
                )

                image_size = torch.tensor(
                    [camera.model.width, camera.model.height], device=sample_uv.device
                )
                new_uv = sample_uv / image_size
                new_uv = torch.logit(new_uv)
                new_cam_ids = torch.full_like(new_uv[:, 0], cam_id, dtype=torch.int32)

                added_cam_ids.append(new_cam_ids)
                added_uv.append(new_uv)
                added_quats.append(new_quats)
                added_depths.append(new_depth)
                added_points.append(new_points)
                added_features_dc.append(new_feat_dc)
                added_features_rest.append(new_feat_rest)

            elif isinstance(self._gaussians, Gaussians3D):
                sample_depth = render_rlt.depth[
                    sample_uv[:, 1].int(), sample_uv[:, 0].int()
                ]
                valid_mask = sample_depth > 0
                sample_depth = sample_depth[valid_mask]
                sample_uv = sample_uv[valid_mask]
                new_points = camera.depth2points(sample_depth, sample_uv, to_world=True)
                new_colors = camera.image[sample_uv[:, 1].int(), sample_uv[:, 0].int()]
                new_feat_dc = RGB2SH(new_colors[:, None, :])
                new_feat_rest = torch.zeros(
                    len(sample_depth),
                    self._gaussians.sh_rest_dim,
                    3,
                    device=new_points.device,
                )
                nn_ret = knn_points(
                    new_points[None, ...], all_old_points[None, ...], K=1
                )
                new_quats = all_old_quats[nn_ret.idx[0, :, 0]]

                added_points.append(new_points)
                added_quats.append(new_quats)
                added_features_dc.append(new_feat_dc)
                added_features_rest.append(new_feat_rest)

        if len(added_points) == 0:
            return

        added_points: torch.Tensor = torch.cat(added_points, dim=0)
        added_quats: torch.Tensor = torch.cat(added_quats, dim=0)
        added_features_dc: torch.Tensor = torch.cat(added_features_dc, dim=0)
        added_features_rest: torch.Tensor = torch.cat(added_features_rest, dim=0)
        # 重新计算scale
        dists = torch.sqrt(distCUDA2(torch.cat([added_points, all_old_points], dim=0)))
        dists = dists[: len(added_points)]
        added_scales = torch.log(dists).reshape(-1, 1).repeat(1, 3)
        added_opacities = torch.logit(torch.ones_like(added_points[:, 0:1]) * 0.1)

        # 重置参数
        print(f"Number of splatters (before densify): {len(self._gaussians)}")
        if isinstance(self._gaussians, GaussiansRayV2):
            added_cam_ids: torch.Tensor = torch.cat(added_cam_ids, dim=0)
            added_uv: torch.Tensor = torch.cat(added_uv, dim=0)
            added_depths: torch.Tensor = torch.cat(added_depths, dim=0)

            new_params = {
                "uvs": added_uv,
                "depths": added_depths.reshape(-1, 1),
                "quats": added_quats,
                "scales": added_scales,
                "opacities": added_opacities,
                "features_dc": added_features_dc,
                "features_rest": added_features_rest,
            }
            self._gaussians.add_parameters(self._optimizer, new_params, added_cam_ids)
        elif isinstance(self._gaussians, Gaussians3D):
            new_params = {
                "means": added_points,
                "quats": added_quats,
                "scales": added_scales,
                "opacities": added_opacities,
                "features_dc": added_features_dc,
                "features_rest": added_features_rest,
            }
            self._gaussians.add_parameters(self._optimizer, new_params)
        print(f"Number of splatters (after densify): {len(self._gaussians)}")

        # reset opacities and scales
        new_opacities = torch.logit(
            torch.ones_like(self._gaussians.get_opacities) * 0.1
        )
        new_dists = torch.sqrt(distCUDA2(self._gaussians.get_means))
        new_scales = torch.log(new_dists).reshape(-1, 1).repeat(1, 3)
        old_scales = self._gaussians.get_scales
        new_scales = torch.where(new_scales < old_scales, new_scales, old_scales)
        self._gaussians.reset_parameters(
            self._optimizer, {"opacities": new_opacities, "scales": new_scales}
        )

    def _init_gaussians_rays(
        self,
        gaussians: GaussiansRayV2,
        new_cam_ids: torch.Tensor,
        new_depths: torch.Tensor,
        new_u: torch.Tensor,
        new_v: torch.Tensor,
        new_colors: torch.Tensor,
        new_dists: torch.Tensor,
    ) -> None:
        # add new parameters for gaussians_ray
        new_scales = torch.log(new_dists).reshape(-1, 1).repeat(1, 3)
        new_quats = generate_random_quats(len(new_depths), self._gaussians.device)
        new_opacities = torch.logit(torch.ones_like(new_depths) * 0.1)
        new_features_dc = RGB2SH(new_colors[..., None, :])
        new_features_rest = torch.zeros(
            [len(new_depths), self._gaussians.sh_rest_dim, 3],
            dtype=torch.float32,
            device=self._gaussians.device,
        )
        new_uv = torch.logit(torch.stack([new_u, new_v], dim=1))

        new_params = {
            "uvs": new_uv,
            "depths": new_depths,
            "quats": new_quats,
            "scales": new_scales,
            "opacities": new_opacities,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
        }
        gaussians.add_parameters(self._optimizer, new_params, new_cam_ids)

    def _init_gaussians_3d(
        self,
        gaussians: Gaussians3D,
        new_points: torch.Tensor,
        new_colors: torch.Tensor,
        new_dists: torch.Tensor,
    ) -> None:
        # add new parameters for gaussians_3d
        new_scales = torch.log(new_dists).reshape(-1, 1).repeat(1, 3)
        new_quats = generate_random_quats(len(new_points), self._gaussians.device)
        new_opacities = torch.logit(torch.ones_like(new_points[:, :1]) * 0.1)
        new_features_dc = RGB2SH(new_colors[..., None, :])
        new_features_rest = torch.zeros(
            [len(new_points), self._gaussians.sh_rest_dim, 3],
            dtype=torch.float32,
            device=self._gaussians.device,
        )

        new_params = {
            "means": new_points,
            "quats": new_quats,
            "scales": new_scales,
            "opacities": new_opacities,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
        }
        gaussians.add_parameters(self._optimizer, new_params)

    @torch.no_grad()
    def init_gaussians_from_prior(self) -> None:

        merged_cam_ids: List[torch.Tensor] = []
        merged_v: List[torch.Tensor] = []
        merged_u: List[torch.Tensor] = []
        merged_colors: List[torch.Tensor] = []
        merged_depths: List[torch.Tensor] = []
        merged_points: List[torch.Tensor] = []

        for cam_id in range(len(self._train_set)):
            camera = self._train_set[cam_id]
            render_rlt: RenderResult = self._renderer(
                self._gaussians.get_render_parameters(self._degree_to_use),
                camera.model,
                camera.w2c,
                self._bg_color,
                need_extra_infos=False,
                require_coord=False,
            )

            color_error: torch.Tensor = torch.abs(render_rlt.image - camera.image)
            color_error = torch.mean(color_error, dim=-1)

            mono_depth = self._priors.get_mono_depth(cam_id)
            mono_depth_mask = self._priors.get_mono_depth_mask(cam_id)

            aligned_mono_depth = mono_depth
            aligned_mono_points = camera.depth2points(aligned_mono_depth)
            self._priors.reset_mono_depth(cam_id, aligned_mono_depth)

            if self._sample_important_pixels:
                sample_mask = sample_alone_edge(
                    camera.image, mono_depth, self._patch_size_for_sample
                )
                # 采样的点必须是高误差的点，且是单目深度合理的点
                sample_mask = mono_depth_mask & sample_mask
            else:
                sample_mask = mono_depth_mask

            sampled_v, sampled_u = torch.where(sample_mask)
            if (
                self._max_num_splatter_per_view > 0
                and len(sampled_v) > self._max_num_splatter_per_view
            ):
                # 如果采样的点数超过最大限制，则随机采样
                rand_idxs = torch.randperm(len(sampled_v), device=sampled_v.device)
                rand_idxs = rand_idxs[: self._max_num_splatter_per_view]
                sampled_v = sampled_v[rand_idxs]
                sampled_u = sampled_u[rand_idxs]
                sample_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
                sample_mask[sampled_v, sampled_u] = True

            sampled_v = (sampled_v.float() + 0.5) / camera.model.height
            sampled_u = (sampled_u.float() + 0.5) / camera.model.width
            sampled_cam_ids = torch.full_like(sampled_v, cam_id, dtype=torch.int32)
            sampled_colors = camera.image[sample_mask]
            sampled_depths = aligned_mono_depth[sample_mask]
            sampled_points = aligned_mono_points[sample_mask]

            merged_cam_ids.append(sampled_cam_ids)
            merged_v.append(sampled_v)
            merged_u.append(sampled_u)
            merged_colors.append(sampled_colors)
            merged_depths.append(sampled_depths)
            merged_points.append(sampled_points)

        # 清空gaussians_ray和gaussians_3d的参数
        self._gaussians.prune_parameters(
            self._optimizer,
            torch.ones(
                len(self._gaussians), dtype=torch.bool, device=self._gaussians.device
            ),
        )

        merged_cam_ids = torch.cat(merged_cam_ids, dim=0)
        merged_v = torch.cat(merged_v, dim=0)
        merged_u = torch.cat(merged_u, dim=0)
        merged_colors = torch.cat(merged_colors, dim=0)
        merged_depths = torch.cat(merged_depths, dim=0)
        merged_points = torch.cat(merged_points, dim=0)
        merged_dists = torch.sqrt(distCUDA2(merged_points))
        print(f"merged_points: {len(merged_points)}")

        # 体素降采样
        if self._downsample_voxel_res > 0:
            min_range = torch.quantile(merged_points, 0.02, dim=0).view(1, 3)
            max_range = torch.quantile(merged_points, 0.98, dim=0).view(1, 3)
            voxel_length = (max_range - min_range) / self._downsample_voxel_res
            voxel_ids = torch.floor(merged_points / voxel_length).int()
            unique_voxel_ids = torch.unique(voxel_ids, dim=0)
            unique_voxel_center = (
                unique_voxel_ids.float() * voxel_length + voxel_length / 2
            )
            nn_ret = knn_points(
                unique_voxel_center[None, ...], merged_points[None, ...], K=1
            )
            nn_idx = nn_ret.idx[0, :, 0]

            # get downsampled points
            new_cam_ids = merged_cam_ids[nn_idx]
            new_u = merged_u[nn_idx]
            new_v = merged_v[nn_idx]
            new_colors = merged_colors[nn_idx]
            new_depths = merged_depths[nn_idx].unsqueeze(-1)
            new_points = merged_points[nn_idx]
            new_dists = torch.sqrt(distCUDA2(new_points))
            print(f"downsampled_points: {len(new_points)}")
        else:
            new_cam_ids = merged_cam_ids
            new_u = merged_u
            new_v = merged_v
            new_colors = merged_colors
            new_depths = merged_depths.unsqueeze(-1)
            new_points = merged_points
            new_dists = merged_dists

        if isinstance(self._gaussians, GaussiansRayV2):
            self._init_gaussians_rays(
                self._gaussians,
                new_cam_ids,
                new_depths,
                new_u,
                new_v,
                new_colors,
                new_dists,
            )
        elif isinstance(self._gaussians, Gaussians3D):
            self._init_gaussians_3d(
                self._gaussians,
                new_points,
                new_colors,
                new_dists,
            )

    @torch.no_grad()
    def __reset_scale(self) -> None:
        scales = torch.exp(self._gaussians.get_scales.clone())
        max_scales = torch.max(scales, dim=1).values
        th1 = torch.quantile(max_scales, 0.95).item()
        th2 = torch.mean(max_scales).item()
        th = min(th1, th2 * 2.0)

        new_scales = torch.log(torch.clamp_max(scales, th))
        self._gaussians.reset_parameters(self._optimizer, {"scales": new_scales})

    def step_after_backward(self, end_points: Dict[str, Any]) -> None:
        pruned = False
        if self._prune_opacity_from <= self._step <= self._prune_opacity_until:
            steps_from_pruning = self._step - self._prune_opacity_from
            if steps_from_pruning % self._prune_opacity_interval == 0:
                self.__prune_low_opacity()
                pruned = True

        if self._step < self._reset_scale_until:
            if self._step % self._reset_scale_interval == 0:
                self.__reset_scale()

        if self._step in self._densify_from_view_at:
            # 每次density之前先删除低透明度的gaussian，减少计算量
            if not pruned:
                self.__prune_low_opacity()
            self.__densify_from_view()
