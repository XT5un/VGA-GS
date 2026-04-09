import os
import random
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
from absl import app
from base_trainer import BaseTrainer
from configs import VGAGSConfigs
from kornia.filters import canny
from origin_trainer import OriginRefinementStrategy
from rich import print
from rich.progress import track
from strategy import VGAGSRefinementStrategy, sample_alone_edge

from scene.data import (
    DataCamera,
    depth2normal,
    get_scene_radius,
    get_scene_scale,
    warp_from_depth,
)
from scene.gaussians import Gaussians3D, GaussiansRayV2
from scene.losses import *
from scene.priors_manager import PriorsManager
from utils import colorize, write_correspondence, write_gray, write_points, write_rgb


class VGAGSTrainer(BaseTrainer):

    def __init__(self) -> None:
        super().__init__()

    def _init_configs(self) -> None:
        self._cfg = VGAGSConfigs()

    def _init_extra_data(self) -> None:
        super()._init_extra_data()
        self._scene_scale = get_scene_scale(self._train_set)

        execute_alignment = True
        self._priors = PriorsManager(
            self._device,
            self._train_set,
            self._cfg.with_background,
            self._cfg.depth_estimation_method,
            self._cfg.image_match_method,
            self._cfg.scene_type,
            self._cfg.use_cache_data,
            execute_alignment,
            bg_color=self._bg_color,
        )

    def _init_models(self) -> None:
        if self._cfg.splatter_type == "3d":
            self._gaussians = Gaussians3D(self._cfg.sh_degree, self._device)
        elif self._cfg.splatter_type == "ray":
            self._gaussians = GaussiansRayV2(self._cfg.sh_degree, self._device)
        else:
            raise ValueError(f"Unknown splatter type: {self._cfg.splatter_type}")

    def _after_init(self) -> None:
        self._cfg.eval_steps = sorted(
            list(set(self._cfg.eval_steps + [self._cfg.max_steps]))
        )
        self._cfg.save_steps = sorted(
            list(set(self._cfg.save_steps + [self._cfg.max_steps]))
        )
        super()._after_init()

    def _set_optimizer(self) -> None:
        params_group = []

        for params in self._gaussians.parameters():
            if params["name"].startswith("depths"):
                params["lr"] = self._cfg.lr_depths
            elif params["name"].startswith("uvs"):
                params["lr"] = self._cfg.lr_uvs
            elif params["name"].startswith("means"):
                params["lr"] = self._cfg.lr_means
            elif params["name"].startswith("scales"):
                params["lr"] = self._cfg.lr_scales
            elif params["name"].startswith("quats"):
                params["lr"] = self._cfg.lr_quats
            elif params["name"].startswith("opacities"):
                params["lr"] = self._cfg.lr_opacities
            elif params["name"].startswith("features_dc"):
                params["lr"] = self._cfg.lr_features
            elif params["name"].startswith("features_rest"):
                params["lr"] = self._cfg.lr_features / 20
            else:
                raise ValueError(f"Unknown parameter: {params['name']}")
            params_group.append(params)

        self._optimizer = torch.optim.Adam(params_group)

    def _set_strategy(self) -> None:
        self._strategy = VGAGSRefinementStrategy(
            self._cfg,
            self._step,
            self._optimizer,
            self._gaussians,
            self._train_set,
            self._priors,
            self._renderer,
            self._bg_color,
        )

    def _set_scheduler(self) -> None:
        self._scheduler = None

    def __init_gaussians_from_priors(
        self, gaussians: GaussiansRayV2 | Gaussians3D
    ) -> None:
        save_dir = os.path.join(self._save_dir, "init_mask")
        os.makedirs(save_dir, exist_ok=True)

        Ks = [camera.model.K for camera in self._train_set]
        poses = [camera.c2w for camera in self._train_set]
        images_size = [
            (camera.model.width, camera.model.height) for camera in self._train_set
        ]
        gaussians.init_cameras(images_size, Ks, poses)

        all_depths, all_colors, all_valid_masks = [], [], []
        for cam_id in range(len(self._train_set)):
            camera = self._train_set[cam_id]
            mono_depth = self._priors.get_mono_depth(cam_id).clone()
            valid_mask = self._priors.get_mono_depth_mask(cam_id).clone()
            valid_mask &= sample_alone_edge(
                camera.image, mono_depth, self._cfg.patch_size_for_sample
            )
            if (
                self._cfg.max_num_splatter_per_view > 0
                and valid_mask.sum().item() > self._cfg.max_num_splatter_per_view
            ):
                valid_V_ids, valid_U_ids = torch.where(valid_mask)
                sample_ids = torch.randperm(
                    len(valid_V_ids), device=valid_V_ids.device
                )[: self._cfg.max_num_splatter_per_view]
                valid_V_ids = valid_V_ids[sample_ids]
                valid_U_ids = valid_U_ids[sample_ids]
                valid_mask = torch.zeros_like(valid_mask)
                valid_mask[valid_V_ids, valid_U_ids] = True
            write_gray(
                os.path.join(save_dir, f"mask_{camera.name}.png"),
                valid_mask.float().cpu().numpy(),
            )

            all_depths.append(mono_depth)
            all_valid_masks.append(valid_mask)
            all_colors.append(camera.image)

        gaussians.init_from_depthmaps(
            all_depths,
            all_colors,
            all_valid_masks,
            optimizer=self._optimizer,
        )

    def _before_train_loop(self) -> None:
        super()._before_train_loop()

        if self._cfg.splatter_type == "ray":
            # init camera parameters
            Ks = [camera.model.K for camera in self._train_set]
            poses = [camera.c2w for camera in self._train_set]
            images_size = [
                (camera.model.width, camera.model.height) for camera in self._train_set
            ]
            self._gaussians.init_cameras(images_size, Ks, poses)

        self._strategy.init_gaussians_from_prior()

        print(f"Number of initilization splatters: {len(self._gaussians)}")

    def _train_step(self) -> Dict[str, Any]:
        end_points = {}
        cam_id = random.randint(0, len(self._train_set) - 1)
        camera = self._train_set[cam_id]
        mono_depth = self._priors.get_mono_depth(cam_id)

        render_rlt = self._renderer(
            self._gaussians.get_render_parameters(self._degree_to_use),
            camera.model,
            camera.w2c,
            self._bg_color,
            need_extra_infos=True,
            require_coord=False,
        )
        end_points["means2d"] = render_rlt.extra["means2d"]
        end_points["radii"] = render_rlt.extra["radii"]
        rendered_image = render_rlt.image
        rendered_depth = render_rlt.depth
        rendered_mid_depth = render_rlt.mid_depth
        rendered_normal = render_rlt.normal
        depth_ratio = self._cfg.fusion_depth_ratio
        if rendered_mid_depth is not None:
            rendered_depth = rendered_mid_depth * depth_ratio + rendered_depth * (
                1 - depth_ratio
            )

        if self._cfg.with_background:
            diff = torch.mean(
                (camera.image - self._bg_color.view(1, 1, 3)) ** 2, dim=-1
            )
            bg_mask = (render_rlt.alpha < 0.3) & (diff < 0.01)
            non_bg_mask = ~bg_mask
        else:
            non_bg_mask = None

        # compute the losses
        loss_l1 = compute_l1_loss(rendered_image, camera.image)
        loss_l1 *= 1 - self._cfg.loss_ssim_lambda
        end_points["loss_l1"] = loss_l1
        loss_ssim = compute_ssim_loss(rendered_image, camera.image)
        loss_ssim *= self._cfg.loss_ssim_lambda
        end_points["loss_ssim"] = loss_ssim

        if self._cfg.loss_group_depth_rank_lambda > 0:
            loss_gdr = compute_group_depth_rank_loss(
                rendered_depth,
                mono_depth,
                self._cfg.loss_group_depth_rank_level,
                mask=non_bg_mask,
            )
            loss_gdr *= self._cfg.loss_group_depth_rank_lambda
            end_points["loss_gdr"] = loss_gdr
        else:
            loss_gdr = 0.0

        if self._cfg.loss_normal_lambda > 0:
            surf_normal = depth2normal(rendered_depth, camera.model)
            if self._cfg.use_mono_depth_normal_consistency:
                mono_normal = depth2normal(mono_depth, camera.model)
                loss_norm1 = compute_normal_loss(
                    rendered_normal, mono_normal, mask=non_bg_mask, ignore_edge=True
                )
                loss_norm1 *= self._cfg.loss_normal_lambda
                end_points["loss_norm1"] = loss_norm1

                loss_norm2 = compute_normal_loss(
                    surf_normal, mono_normal, mask=non_bg_mask, ignore_edge=True
                )
                loss_norm2 *= self._cfg.loss_normal_lambda
                end_points["loss_norm2"] = loss_norm2
            else:
                loss_norm1 = compute_normal_loss(
                    rendered_normal, surf_normal, mask=non_bg_mask, ignore_edge=True
                )
                loss_norm1 *= self._cfg.loss_normal_lambda
                end_points["loss_norm1"] = loss_norm1

                loss_norm2 = 0.0
        else:
            loss_norm1 = loss_norm2 = 0.0

        # if self._cfg.splatter_type == "ray":
        #     if isinstance(self._gaussians, GaussiansRayV2):
        #         gs_ray = self._gaussians
        #     elif isinstance(self._gaussians, GaussiansMixed):
        #         gs_ray = self._gaussians.gaussians_ray
        #     if gs_ray.num_cameras > 0:
        #         sparse_depths, sparse_uvs = gs_ray.get_sub_sparse_depths(cam_id)
        #         if len(sparse_depths) == 0:
        #             loss_close = 0.0
        #         else:
        #             close_depth_mask = (
        #                 sparse_depths < self._cfg.loss_depth_close_threshold
        #             )
        #             if close_depth_mask.any():
        #                 loss_close = torch.mean(1.0 / sparse_depths[close_depth_mask])
        #                 loss_close *= self._cfg.loss_depth_close_lambda
        #                 end_points["loss_close"] = loss_close
        #             else:
        #                 loss_close = 0.0
        #     else:
        #         loss_close = 0.0
        # else:
        #     loss_close = 0.0

        loss = loss_l1 + loss_ssim + loss_norm1 + loss_gdr + loss_norm2

        if (
            self._step < self._cfg.train_pseudo_view_from
            or self._step > self._cfg.train_pseudo_view_until
            or self._step % self._cfg.train_pseudo_view_interval != 0
        ):
            end_points["loss"] = loss
            return end_points

        # pseudo view regularization
        if self._cfg.warp_midian_depth and render_rlt.mid_depth is not None:
            rendered_depth = render_rlt.mid_depth
        else:
            rendered_depth = render_rlt.depth
        rendered_depth = rendered_depth.detach()

        p_camera = camera.create_pseudo_camera(
            self._cfg.pseudo_trans_range * self._scene_scale,
            self._cfg.pseudo_angle_range,
        )
        p_render_rlt = self._renderer(
            self._gaussians.get_render_parameters(self._degree_to_use),
            p_camera.model,
            p_camera.w2c,
            self._bg_color,
            need_extra_infos=True,
        )

        # warp frame from p_view to current view
        p_noraml = p_render_rlt.normal
        p_surf_normal = depth2normal(p_render_rlt.depth, p_camera.model)
        pseudo2train_rotmat = camera.w2c[:3, :3] @ p_camera.c2w[:3, :3]
        p_noraml_in_train = torch.einsum("ij,...j->...i", pseudo2train_rotmat, p_noraml)
        p_surf_normal_in_train = torch.einsum(
            "ij,...j->...i", pseudo2train_rotmat, p_surf_normal
        )
        stacked_feats = torch.cat(
            [
                p_render_rlt.image,
                p_render_rlt.depth[..., None],
                p_noraml_in_train,
                p_surf_normal_in_train,
            ],
            dim=-1,
        )
        warped_feats = warp_from_depth(stacked_feats, rendered_depth, p_camera, camera)
        warped_image = warped_feats[..., :3]
        warped_depth = warped_feats[..., 3]
        warped_normal = warped_feats[..., 4:7]
        warped_surf_normal = warped_feats[..., 7:]

        vis_mask = warped_depth > 0
        if self._cfg.with_background:
            vis_mask = vis_mask & non_bg_mask
        image_gt = camera.image.clone()
        image_gt[~vis_mask] = 0.0
        warped_image[~vis_mask] = 0.0
        loss_w_l1 = compute_l1_loss(warped_image[vis_mask], image_gt[vis_mask])
        loss_w_l1 *= 1 - self._cfg.loss_ssim_lambda
        loss_w_l1 *= self._cfg.loss_pseudo_view_weight
        loss_w_ssim = compute_ssim_loss(warped_image, image_gt)
        loss_w_ssim *= self._cfg.loss_ssim_lambda
        loss_w_ssim *= self._cfg.loss_pseudo_view_weight
        end_points["loss_w_l1"] = loss_w_l1
        end_points["loss_w_ssim"] = loss_w_ssim

        if self._cfg.loss_group_depth_rank_lambda > 0:
            loss_w_gdr = compute_group_depth_rank_loss(
                warped_depth,
                mono_depth,
                self._cfg.loss_group_depth_rank_level,
                mask=vis_mask,
            )
            loss_w_gdr *= self._cfg.loss_group_depth_rank_lambda
            loss_w_gdr *= self._cfg.loss_pseudo_view_weight
            end_points["loss_w_gdr"] = loss_w_gdr
        else:
            loss_w_gdr = 0.0

        if self._cfg.loss_normal_lambda > 0:
            if self._cfg.use_mono_depth_normal_consistency:
                loss_w_norm1 = compute_normal_loss(
                    warped_normal, mono_normal, mask=vis_mask, ignore_edge=True
                )
                loss_w_norm1 *= self._cfg.loss_normal_lambda
                loss_w_norm1 *= self._cfg.loss_pseudo_view_weight
                end_points["loss_w_norm1"] = loss_w_norm1

                loss_w_norm2 = compute_normal_loss(
                    warped_surf_normal, mono_normal, mask=vis_mask, ignore_edge=True
                )
                loss_w_norm2 *= self._cfg.loss_normal_lambda
                loss_w_norm2 *= self._cfg.loss_pseudo_view_weight
                end_points["loss_w_norm2"] = loss_w_norm2
            else:
                loss_w_norm1 = compute_normal_loss(
                    warped_normal, warped_surf_normal, mask=vis_mask, ignore_edge=True
                )
                loss_w_norm1 *= self._cfg.loss_normal_lambda
                loss_w_norm1 *= self._cfg.loss_pseudo_view_weight
                end_points["loss_w_norm1"] = loss_w_norm1

                loss_w_norm2 = 0.0
        else:
            loss_w_norm1 = loss_w_norm2 = 0.0

        loss += loss_w_l1 + loss_w_ssim + loss_w_gdr + loss_w_norm1 + loss_w_norm2

        end_points["loss"] = loss
        return end_points


def main(argv) -> None:
    trainer = VGAGSTrainer()
    trainer.train()


if __name__ == "__main__":
    app.run(main)
