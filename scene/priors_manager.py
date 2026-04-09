import os
import random
from itertools import combinations
from typing import Any, Dict, List, Literal, Set, Tuple, TypeAlias

import kornia.geometry.epipolar as epipolar
import numpy as np
import torch
from networkx import Graph
from rich import print
from rich.progress import track
from torch import Tensor

from utils import (
    clean_depthmap,
    colorize,
    write_correspondence,
    write_points,
    write_rgb,
)

from .data.dataset import Dataset
from .data.priors import *

SUPPORTED_DEPTH_METHODS: TypeAlias = Literal[
    "Metric3D",
    "DA",
    "DepthPro",
]


class PriorsManager:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        with_background: bool = False,
        depth_method: SUPPORTED_DEPTH_METHODS = "DA",
        match_method: Literal["DKM"] = "DKM",
        scene_type: Literal["indoor", "outdoor"] = "indoor",
        use_cache_data: bool = True,
        execute_alignment: bool = False,
        corase_resolution: int | Tuple[int, int] = 1600,
        vis_priors: bool = True,
        bg_color: Tensor | None = None,
    ) -> None:
        self._device = device
        self._use_cache_data = use_cache_data
        self._vis_priors = vis_priors
        self._with_background = with_background
        self._bg_color = (
            bg_color if bg_color is not None else torch.zeros(3, device=device)
        )
        self._color_eps = 0.01

        if isinstance(corase_resolution, int):
            corase_resolution = (corase_resolution, corase_resolution)

        image_dict = self._dataset_prepocess(dataset)
        image_dict = self._predict_mono_depths(
            image_dict,
            dataset.data_dir,
            depth_method,
            scene_type,
        )

        if execute_alignment:
            correspondence_dict = self._predict_correspondences(
                image_dict, dataset.data_dir, match_method
            )

            image_dict = self._align_mono_depth(
                image_dict, correspondence_dict, corase_resolution
            )

        self._mono_depths: List[Tensor] = []
        self._valid_mono_depths: List[Tensor] = []
        for cam_id in range(len(dataset)):
            camera = dataset[cam_id]
            name = camera.name
            self._mono_depths.append(image_dict[name]["depth"])
            self._valid_mono_depths.append(image_dict[name]["depth_mask"])

    def reset_mono_depth(self, camera_id: int, new_mono_depth: Tensor) -> None:
        self._mono_depths[camera_id] = new_mono_depth

    def get_mono_depth(self, cam_id: int) -> Tensor:
        return self._mono_depths[cam_id]

    def get_mono_depth_mask(self, cam_id: int) -> Tensor:
        return self._valid_mono_depths[cam_id]

    def _dataset_prepocess(self, dataset: Dataset) -> Dict[str, Dict[str, Any]]:
        processed_dict = {}
        for cam_id in track(range(len(dataset)), description="Preprocess dataset"):
            camera = dataset[cam_id]
            name = camera.name
            image = camera.image.clone()
            pose = camera.c2w.clone()
            K = camera.model.K.clone()
            H, W = image.shape[:2]

            camera_info = {
                "image": image,
                "pose": pose,
                "intrinsic": K,
            }

            processed_dict[name] = camera_info

        return processed_dict

    def _get_mono_depth_from_model(
        self, model: BaseDepthEstimator | None, image: Tensor, name: str, prior_dir: str
    ) -> Tensor:
        H, W = image.shape[:2]
        depth_path = os.path.join(prior_dir, f"{name}_{H}x{W}.npy")
        if self._use_cache_data and os.path.exists(depth_path):
            depth = np.load(depth_path)
            depth = torch.tensor(depth, dtype=torch.float32, device=self._device)
        else:
            assert model is not None
            depth = model.predict(image)

        if self._use_cache_data and not os.path.exists(depth_path):
            np.save(depth_path, depth.cpu().numpy())

        if self._vis_priors:
            vis_depth_path = os.path.join(prior_dir, f"{name}_{H}x{W}.jpg")
            if not os.path.exists(vis_depth_path):
                write_rgb(vis_depth_path, colorize(depth.cpu().numpy()))

        torch.cuda.empty_cache()
        return depth

    def _predict_mono_depths(
        self,
        image_dict: Dict[str, Dict[str, Any]],
        data_dir: str,
        method_name: SUPPORTED_DEPTH_METHODS,
        scene_type: Literal["indoor", "outdoor"],
        sky_depth_threshold: float = 75.0,
    ) -> Dict[str, Dict[str, Any]]:

        if method_name == "Metric3D":
            model = Metric3DModel(self._device)
        elif method_name == "DA":
            model = DepthAnythingModel(self._device, scene_type=scene_type)
        elif method_name == "DepthPro":
            model = DepthProModel(self._device)
        elif method_name in ("UniDepth", "dust3r", "vggt"):
            model = None
            assert self._use_cache_data, f"{method_name} must use pre-computed depths."
        else:
            raise ValueError(f"Unsupported depth estimation method: {method_name}")

        prior_dir = os.path.join(
            data_dir, "priors", "depth_estimation", f"{method_name}"
        )
        if self._use_cache_data and not os.path.exists(prior_dir):
            os.makedirs(prior_dir)

        for name in track(image_dict.keys(), "Generate Mono-depth"):
            image: Tensor = image_dict[name]["image"]
            intrinsic: Tensor = image_dict[name]["intrinsic"]
            depth = self._get_mono_depth_from_model(model, image, name, prior_dir)

            _, depth_mask = clean_depthmap(depth.cpu().numpy(), intrinsic.cpu().numpy())
            depth_mask = torch.tensor(depth_mask, dtype=torch.bool, device=self._device)

            # filter invalid area
            sky_mask = depth > sky_depth_threshold
            depth_mask = depth_mask & ~sky_mask
            if self._with_background:
                diff = torch.mean((image - self._bg_color[None, None, :]) ** 2, dim=-1)
                bg_color_mask = diff < self._color_eps

                if bg_color_mask.any():
                    bg_depth_th = torch.quantile(depth[bg_color_mask], 0.02)
                    bg_mask = (depth > bg_depth_th) & bg_color_mask
                    depth_mask = depth_mask & ~bg_mask

            image_dict[name]["depth"] = depth
            image_dict[name]["depth_mask"] = depth_mask

        del model
        torch.cuda.empty_cache()

        return image_dict

    def _get_correspondences_from_model(
        self,
        model: BaseMatcher,
        image0: Tensor,
        image1: Tensor,
        name0: str,
        name1: str,
        prior_dir: str,
    ) -> Tuple[Tensor, Tensor]:
        H0, W0 = image0.shape[:2]
        H1, W1 = image1.shape[:2]
        corr_path = os.path.join(prior_dir, f"{name0}_{H0}x{W0}-{name1}_{H1}x{W1}.npy")
        if self._use_cache_data and os.path.exists(corr_path):
            corr = np.load(corr_path)
            kpts0 = torch.tensor(corr[:, :2], dtype=torch.float32, device=self._device)
            kpts1 = torch.tensor(corr[:, 2:], dtype=torch.float32, device=self._device)
        else:
            kpts0, kpts1 = model.predict(image0, image1, filter_by_fundamental=False)

        if self._use_cache_data and not os.path.exists(corr_path):
            corr = torch.cat([kpts0, kpts1], dim=1).cpu().numpy()
            np.save(corr_path, corr)

        if self._vis_priors:
            vis_corr_path = os.path.join(
                prior_dir, f"{name0}_{H0}x{W0}-{name1}_{H1}x{W1}.jpg"
            )
            if not os.path.exists(vis_corr_path):
                write_correspondence(
                    vis_corr_path,
                    image0.cpu().numpy(),
                    image1.cpu().numpy(),
                    kpts0.cpu().numpy(),
                    kpts1.cpu().numpy(),
                )

        torch.cuda.empty_cache()
        return kpts0, kpts1

    def _predict_correspondences(
        self,
        image_dict: Dict[str, Dict[str, Any]],
        data_dir: str,
        method_name: Literal["DKM"],
        min_num_matches: int = 50,
        epipolar_threshold: float = 4.0,
        inlier_ratio: float = 0.5,
    ) -> Dict[Tuple[str, str], Tuple[Tensor, Tensor]]:
        if method_name == "DKM":
            model = DKMModel(self._device)
        else:
            raise ValueError(f"Unsupported image matching method: {method_name}")

        prior_dir = os.path.join(data_dir, "priors", "image_matching", f"{method_name}")
        if self._use_cache_data and not os.path.exists(prior_dir):
            os.makedirs(prior_dir)

        correspondence_dict: Dict[Tuple[str, str], Tuple[Tensor, Tensor]] = {}
        pairs = list(combinations(sorted(image_dict.keys()), 2))
        for name0, name1 in track(pairs, "Image Matching"):
            image0 = image_dict[name0]["image"]
            image1 = image_dict[name1]["image"]
            kpts0, kpts1 = self._get_correspondences_from_model(
                model, image0, image1, name0, name1, prior_dir
            )
            if len(kpts0) < min_num_matches:
                continue

            # filter by epipolar constraint
            w2c0_3x4 = torch.inverse(image_dict[name0]["pose"])[:3, :]
            w2c1_3x4 = torch.inverse(image_dict[name1]["pose"])[:3, :]
            prj_mat0 = image_dict[name0]["intrinsic"] @ w2c0_3x4
            prj_mat1 = image_dict[name1]["intrinsic"] @ w2c1_3x4
            f_mat = epipolar.fundamental_from_projections(prj_mat0, prj_mat1)
            e_dist01 = epipolar.left_to_right_epipolar_distance(
                kpts0[None, ...], kpts1[None, ...], f_mat[None, ...]
            )[0]
            e_dist10 = epipolar.right_to_left_epipolar_distance(
                kpts0[None, ...], kpts1[None, ...], f_mat[None, ...]
            )[0]
            e_dist = torch.maximum(e_dist01, e_dist10)
            inlier_mask = e_dist < epipolar_threshold
            inlier_num = inlier_mask.sum().item()
            if (
                inlier_num / len(inlier_mask) < inlier_ratio
                or inlier_num < min_num_matches
            ):
                continue

            kpts0 = kpts0[inlier_mask]
            kpts1 = kpts1[inlier_mask]

            valid_mask0 = image_dict[name0]["depth_mask"]
            valid_mask1 = image_dict[name1]["depth_mask"]
            kpts0_mask = valid_mask0[kpts0[:, 1].long(), kpts0[:, 0].long()]
            kpts1_mask = valid_mask1[kpts1[:, 1].long(), kpts1[:, 0].long()]
            kpts_mask = kpts0_mask & kpts1_mask
            kpts0 = kpts0[kpts_mask]
            kpts1 = kpts1[kpts_mask]

            correspondence_dict[(name0, name1)] = (kpts0, kpts1)

        return correspondence_dict

    def _get_low_res_data(
        self,
        image_dict: Dict[str, Dict[str, Any]],
        corr_dict: Dict[Tuple[str, str], Tuple[Tensor, Tensor]],
        low_res: Tuple[int, int],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Tuple[Tensor, Tensor]]]:
        low_res_image_dict = {}
        factors = {}
        for name in image_dict:
            raw_image: Tensor = image_dict[name]["image"]
            H, W = raw_image.shape[:2]
            if H * W <= low_res[0] * low_res[1]:
                low_res_image_dict[name] = {
                    "image": raw_image,
                    "depth": image_dict[name]["depth"],
                    "intrinsic": image_dict[name]["intrinsic"],
                    "pose": image_dict[name]["pose"],
                    "depth_mask": image_dict[name]["depth_mask"],
                }
                factors[name] = None
                continue

            factor = np.sqrt((low_res[0] * low_res[1]) / (H * W))
            low_res_H, low_res_W = int(H * factor), int(W * factor)
            low_res_image = torch.nn.functional.interpolate(
                raw_image[None, ...].permute(0, 3, 1, 2),
                size=(low_res_H, low_res_W),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)
            low_res_depth = torch.nn.functional.interpolate(
                image_dict[name]["depth"][None, None, ...],
                size=(low_res_H, low_res_W),
                mode="nearest",
            )[0, 0]
            low_res_depth_mask = torch.nn.functional.interpolate(
                image_dict[name]["depth_mask"][None, None, ...].float(),
                size=(low_res_H, low_res_W),
                mode="nearest",
            )[0, 0]
            low_res_depth_mask = low_res_depth_mask > 0.9
            low_res_intrinsic = image_dict[name]["intrinsic"].clone()
            low_res_intrinsic[:2, :2] *= factor
            low_res_pose = image_dict[name]["pose"].clone()
            low_res_image_dict[name] = {
                "image": low_res_image,
                "depth": low_res_depth,
                "intrinsic": low_res_intrinsic,
                "pose": low_res_pose,
                "depth_mask": low_res_depth_mask,
            }
            factors[name] = factor

        low_res_corr_dict = {}
        for name0, name1 in corr_dict.keys():
            kpts0, kpts1 = corr_dict[(name0, name1)]
            if factors[name0] is not None:
                kpts0 = kpts0 * factors[name0]
            if factors[name1] is not None:
                kpts1 = kpts1 * factors[name1]
            low_res_corr_dict[(name0, name1)] = (kpts0, kpts1)

        return low_res_image_dict, low_res_corr_dict

    def _coarse_align_mono_depth(
        self,
        image_dict: Dict[str, Dict[str, Any]],
        corr_dict: Dict[Tuple[str, str], Tuple[Tensor, Tensor]],
    ):
        def _optimize_single_view_scale(ref_name: str, sup_names: List[str]) -> float:
            name0 = ref_name
            scale = torch.tensor(
                1.0, dtype=torch.float32, device=self._device, requires_grad=True
            )
            depth0 = image_dict[name0]["depth"]
            optimizer = torch.optim.Adam([scale], lr=0.05)
            for iter in track(range(200), f"Coarse-align with {name0}"):
                all_prj_pixels, all_gt_pixels = [], []
                for name1 in sup_names:
                    if (name0, name1) not in corr_dict:
                        kpts1, kpts0 = corr_dict[(name1, name0)]
                    else:
                        kpts0, kpts1 = corr_dict[(name0, name1)]

                    cam0_to_world = image_dict[name0]["pose"]
                    cam1_to_world = image_dict[name1]["pose"]
                    cam0_to_cam1 = torch.inverse(cam1_to_world) @ cam0_to_world
                    K0_inv = torch.inverse(image_dict[name0]["intrinsic"])
                    K1 = image_dict[name1]["intrinsic"]

                    u0, v0 = kpts0[:, 0], kpts0[:, 1]
                    d0 = depth0[v0.long(), u0.long()] * scale
                    pts0 = (
                        torch.stack([u0, v0, torch.ones_like(u0)], dim=1) * d0[:, None]
                    )
                    pts0 = torch.einsum("ij,nj->ni", K0_inv, pts0)
                    pts0 = torch.einsum(
                        "ij,nj->ni", cam0_to_cam1[:3, :3], pts0
                    ) + cam0_to_cam1[:3, 3].view(1, 3)
                    prj0 = torch.einsum("ij,nj->ni", K1, pts0)
                    prj_depth0 = prj0[:, 2]
                    prj0 = prj0[:, :2] / prj_depth0[:, None]

                    all_prj_pixels.append(prj0)
                    all_gt_pixels.append(kpts1)

                all_prj_pixels = torch.cat(all_prj_pixels, dim=0)
                all_gt_pixels = torch.cat(all_gt_pixels, dim=0)
                loss = torch.norm(all_prj_pixels - all_gt_pixels, dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if iter % 10 == 0:
                #     print(f"iter: {iter}, loss: {loss.item()}, scale: {scale.item()}")

            return scale.item()

        covis_graph = Graph()
        for name in image_dict:
            covis_graph.add_node(name, num_views=0, num_matches=0)

        for origin_name, name1 in corr_dict.keys():
            kpts0, kpts1 = corr_dict[(origin_name, name1)]
            covis_graph.nodes[origin_name]["num_views"] += 1
            covis_graph.nodes[name1]["num_views"] += 1
            covis_graph.nodes[origin_name]["num_matches"] += len(kpts0)
            covis_graph.nodes[name1]["num_matches"] += len(kpts1)
            covis_graph.add_edge(origin_name, name1)

        # 找出共视最多的视图，如果共视视图一样多就找出匹配最多的视图
        origin_name = max(
            covis_graph.nodes(data=True),
            key=lambda x: (x[1]["num_views"], x[1]["num_matches"]),
        )[0]
        scale = _optimize_single_view_scale(
            origin_name, list(covis_graph.neighbors(origin_name))
        )

        for name in covis_graph.nodes:
            image_dict[name]["depth"] *= scale

        return image_dict

    def _align_mono_depth(
        self,
        image_dict: Dict[str, Dict[str, Any]],
        corr_dict: Dict[Tuple[str, str], Tuple[Tensor, Tensor]],
        low_res: Tuple[int, int],
        max_iters: int = 2000,
    ) -> Dict[str, Dict[str, Any]]:
        # pre-align mono-depths is important for scale-unknown datasets
        self._coarse_align_mono_depth(image_dict, corr_dict)

        low_res_images_dict, low_res_corrs_dict = self._get_low_res_data(
            image_dict, corr_dict, low_res
        )

        name2idx = {}
        mono_depths = []
        valid_masks = []
        colors = []
        Ks = []
        c2ws = []
        for idx, name in enumerate(low_res_images_dict):
            name2idx[name] = idx
            mono_depths.append(low_res_images_dict[name]["depth"])
            valid_masks.append(low_res_images_dict[name]["depth_mask"])
            colors.append(low_res_images_dict[name]["image"])
            Ks.append(low_res_images_dict[name]["intrinsic"])
            c2ws.append(low_res_images_dict[name]["pose"])
        mono_depths = torch.stack(mono_depths, dim=0)
        valid_masks = torch.stack(valid_masks, dim=0)
        colors = torch.stack(colors, dim=0)
        Ks = torch.stack(Ks, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        all_kpts0_idxs, all_kpts1_idxs = [], []
        for name0, name1 in low_res_corrs_dict:
            H0, W0 = low_res_images_dict[name0]["depth"].shape[:2]
            H1, W1 = low_res_images_dict[name1]["depth"].shape[:2]
            U0 = low_res_corrs_dict[(name0, name1)][0][:, 0]
            V0 = low_res_corrs_dict[(name0, name1)][0][:, 1]
            U1 = low_res_corrs_dict[(name0, name1)][1][:, 0]
            V1 = low_res_corrs_dict[(name0, name1)][1][:, 1]
            U0_idx = torch.clamp(U0.long(), 0, W0 - 1)
            V0_idx = torch.clamp(V0.long(), 0, H0 - 1)
            U1_idx = torch.clamp(U1.long(), 0, W1 - 1)
            V1_idx = torch.clamp(V1.long(), 0, H1 - 1)

            batch_idx0 = torch.full_like(U0_idx, name2idx[name0])
            batch_idx1 = torch.full_like(U1_idx, name2idx[name1])
            all_kpts0_idxs.append(torch.stack([batch_idx0, V0_idx, U0_idx], dim=1))
            all_kpts1_idxs.append(torch.stack([batch_idx1, V1_idx, U1_idx], dim=1))

            # # ! debug
            # d0 = low_res_images_dict[name0]["depth"][V0_idx, U0_idx]
            # pts0 = torch.stack([U0, V0, torch.ones_like(U0)], dim=1) * d0[:, None]
            # pts0 = (torch.inverse(low_res_images_dict[name0]["intrinsic"]) @ pts0.T).T
            # rgb0 = low_res_images_dict[name0]["image"][V0_idx, U0_idx]
            # write_points(
            #     f"{name0}-{name1}_kpts3d0.ply", pts0.cpu().numpy(), rgb0.cpu().numpy()
            # )
            # d1 = low_res_images_dict[name1]["depth"][V1_idx, U1_idx]
            # pts1 = torch.stack([U1, V1, torch.ones_like(U1)], dim=1) * d1[:, None]
            # pts1 = (torch.inverse(low_res_images_dict[name1]["intrinsic"]) @ pts1.T).T
            # rgb1 = low_res_images_dict[name1]["image"][V1_idx, U1_idx]
            # write_points(
            #     f"{name0}-{name1}_kpts3d1.ply", pts1.cpu().numpy(), rgb1.cpu().numpy()
            # )

        all_kpts0_idxs = torch.cat(all_kpts0_idxs, dim=0)
        all_kpts1_idxs = torch.cat(all_kpts1_idxs, dim=0)

        uu, vv = torch.meshgrid(
            torch.arange(mono_depths.shape[2]),
            torch.arange(mono_depths.shape[1]),
            indexing="xy",
        )
        uu = uu.float().to(self._device) + 0.5
        vv = vv.float().to(self._device) + 0.5
        grid = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)  # (H, W, 3)
        grid = grid[None, ...]  # (1, H, W, 3)
        Ks_inv = torch.inverse(Ks)  # (N, 3, 3)
        rot = c2ws[:, :3, :3]  # (N, 3, 3)
        trans = c2ws[:, :3, 3].view(-1, 1, 1, 3)  # (N, 3)

        depth_scale = torch.ones(
            (len(mono_depths), 1, 1),
            dtype=torch.float32,
            device=self._device,
            requires_grad=True,
        )
        depth_shift = torch.zeros(
            (len(mono_depths), 1, 1),
            dtype=torch.float32,
            device=self._device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([depth_scale, depth_shift], lr=0.02)

        for iter in track(range(max_iters), "Align Mono-depth"):
            mono_depths_aligned = mono_depths * depth_scale + depth_shift
            cam_pts3d = torch.einsum(
                "nij,nhwj->nhwi", Ks_inv, grid * mono_depths_aligned[..., None]
            )
            world_pts3d = torch.einsum("nij,nhwj->nhwi", rot, cam_pts3d) + trans

            kpts3d0 = world_pts3d[
                all_kpts0_idxs[:, 0], all_kpts0_idxs[:, 1], all_kpts0_idxs[:, 2]
            ]
            kpts3d1 = world_pts3d[
                all_kpts1_idxs[:, 0], all_kpts1_idxs[:, 1], all_kpts1_idxs[:, 2]
            ]
            kpts3d_dist = torch.norm(kpts3d0 - kpts3d1, dim=-1)
            loss = kpts3d_dist.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        depth_scale = depth_scale.detach()
        depth_shift = depth_shift.detach()
        for idx, name in enumerate(low_res_images_dict):
            depth = image_dict[name]["depth"]
            valid_mask = depth > 0
            depth = depth * depth_scale[idx] + depth_shift[idx]
            depth[~valid_mask] = 0.0
            image_dict[name]["depth"] = depth

        torch.cuda.empty_cache()
        return image_dict
