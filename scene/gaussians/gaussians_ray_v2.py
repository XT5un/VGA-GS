from typing import Dict, List, Tuple

import numpy as np
import pypose as pp
import torch
from simple_knn._C import distCUDA2
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from .base_gaussians import BaseGaussians, RenderGaussiansParams, STDGaussiansParams
from .utils import RGB2SH, generate_random_quats


def _depth2camera_points(
    depths: Tensor,
    u: Tensor,
    v: Tensor,
    fx: Tensor,
    fy: Tensor,
    cx: Tensor,
    cy: Tensor,
    indices: Tensor,
) -> Tensor:
    fx = fx[indices]
    fy = fy[indices]
    cx = cx[indices]
    cy = cy[indices]

    depths = depths.squeeze(1)
    x = (u - cx) * depths / fx
    y = (v - cy) * depths / fy
    z = depths
    return torch.stack([x, y, z], dim=1)


def _camera_points2world_points(
    points: Tensor,
    poses: pp.LieTensor,
    indices: Tensor,
) -> Tensor:
    if len(indices) == 0:
        return points

    poses = poses[indices]
    world_points = poses @ points
    return world_points


class GaussiansRayV2(BaseGaussians):
    def __init__(
        self,
        sh_degree: int = 3,
        device: str | torch.device = "cuda",
        suffix: str | None = None,
    ) -> None:
        super().__init__(sh_degree, device, suffix)
        self._camera_indices = torch.empty(0, dtype=torch.int32, device=self._device)
        self._fx = torch.empty(0, dtype=torch.float32, device=self._device)
        self._fy = torch.empty(0, dtype=torch.float32, device=self._device)
        self._cx = torch.empty(0, dtype=torch.float32, device=self._device)
        self._cy = torch.empty(0, dtype=torch.float32, device=self._device)
        self._image_size = torch.empty(0, 2, dtype=torch.float32, device=self._device)
        self._poses = pp.SE3(
            torch.empty(0, 7, dtype=torch.float32, device=self._device)
        )

    @property
    def num_cameras(self) -> int:
        return len(self._poses)

    def __len__(self) -> int:
        return len(self._params_dict[f"depths{self._suffix}"])

    def _build_parameters(self) -> Dict[str, Parameter]:
        uvs = torch.empty(0, 2, dtype=torch.float32, device=self._device)
        depths = torch.empty(0, 1, dtype=torch.float32, device=self._device)
        quats = torch.empty(0, 4, dtype=torch.float32, device=self._device)
        scales = torch.empty(0, 3, dtype=torch.float32, device=self._device)
        opacities = torch.empty(0, 1, dtype=torch.float32, device=self._device)
        features_dc = torch.empty(0, 1, 3, dtype=torch.float32, device=self._device)
        features_rest = torch.empty(
            0, self._sh_rest_dim, 3, dtype=torch.float32, device=self._device
        )

        return {
            f"uvs{self._suffix}": Parameter(uvs, requires_grad=True),
            f"depths{self._suffix}": Parameter(depths, requires_grad=True),
            f"quats{self._suffix}": Parameter(quats, requires_grad=True),
            f"scales{self._suffix}": Parameter(scales, requires_grad=True),
            f"opacities{self._suffix}": Parameter(opacities, requires_grad=True),
            f"features_dc{self._suffix}": Parameter(features_dc, requires_grad=True),
            f"features_rest{self._suffix}": Parameter(
                features_rest, requires_grad=True
            ),
        }

    @property
    def get_means(self) -> Tensor:
        return self.__get_means3D()

    def __get_means3D(self, camera_id: int | None = None) -> Tensor:
        camera_indices = self._camera_indices
        d = self._params_dict[f"depths{self._suffix}"]
        uvs = self._params_dict[f"uvs{self._suffix}"]
        if camera_id is not None:
            mask = camera_indices == camera_id
            camera_indices = camera_indices[mask]
            d = d[mask]
            uvs = uvs[mask]
        image_size = self._image_size[camera_indices]
        uvs = torch.sigmoid(uvs) * image_size

        cam_points = _depth2camera_points(
            d,
            uvs[..., 0],
            uvs[..., 1],
            self._fx,
            self._fy,
            self._cx,
            self._cy,
            camera_indices,
        )
        world_points = _camera_points2world_points(
            cam_points, self._poses, camera_indices
        )
        return world_points

    def get_std_parameters(self) -> STDGaussiansParams:
        means = self.__get_means3D()
        return STDGaussiansParams(
            means=means,
            quats=self._params_dict[f"quats{self._suffix}"],
            scales=self._params_dict[f"scales{self._suffix}"],
            opacities=self._params_dict[f"opacities{self._suffix}"],
            features_dc=self._params_dict[f"features_dc{self._suffix}"],
            features_rest=self._params_dict[f"features_rest{self._suffix}"],
        )

    def get_sub_sparse_depths(
        self, camera_id: int, return_mask: bool = False
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        assert camera_id < self.num_cameras
        mask = self._camera_indices == camera_id
        depths = self._params_dict[f"depths{self._suffix}"][mask].squeeze(-1)
        uvs = torch.sigmoid(self._params_dict[f"uvs{self._suffix}"][mask])
        uvs = uvs * self._image_size[camera_id].view(1, 2)

        if return_mask:
            return depths, uvs, mask

        return depths, uvs

    def init_cameras(
        self,
        images_size: List[Tuple[int, int]] | Tensor,
        Ks: List[Tensor] | Tensor,
        poses: List[Tensor] | Tensor,
    ) -> None:
        if isinstance(images_size, list):
            images_size = torch.tensor(
                images_size, dtype=torch.float32, device=self._device
            )
        else:
            images_size = images_size.to(self._device)
        assert images_size.shape[1] == 2

        if isinstance(Ks, list):
            Ks = torch.stack(Ks, dim=0).to(self._device)
        else:
            Ks = Ks.to(self._device)
        assert Ks.shape[1:] == (3, 3)

        if isinstance(poses, list):
            poses = torch.stack(poses, dim=0).to(self._device)
        else:
            poses = poses.to(self._device)
        assert poses.shape[1:] == (4, 4)

        assert len(images_size) == len(Ks) == len(poses)
        self._fx = Ks[:, 0, 0]
        self._fy = Ks[:, 1, 1]
        self._cx = Ks[:, 0, 2]
        self._cy = Ks[:, 1, 2]
        self._image_size = images_size
        self._poses = pp.from_matrix(poses, ltype=pp.SE3_type, check=False)

    def init_from_depthmaps(
        self,
        depthmaps: List[torch.Tensor],
        colors: List[torch.Tensor | None] = None,
        valid_masks: List[torch.Tensor | None] = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        assert self.num_cameras > 0, "Cameras must be initialized first."
        assert len(depthmaps) == self.num_cameras, "Number of cameras mismatch."

        if colors is None:
            colors = [None] * self.num_cameras
        if valid_masks is None:
            valid_masks = [None] * self.num_cameras

        depths = []
        indices = []
        Us, Vs = [], []
        valid_colors = []
        for cam_id in range(self.num_cameras):
            depth_map = depthmaps[cam_id].to(self._device)
            valid_mask = valid_masks[cam_id]
            if valid_mask is None:
                valid_mask = torch.ones_like(depth_map, dtype=torch.bool)
            else:
                valid_mask = valid_mask.to(self._device)
            v, u = torch.where(valid_mask)
            u = u.float() + 0.5
            v = v.float() + 0.5

            depth = depth_map[valid_mask].reshape(-1)
            depths.append(depth)
            indices.append(torch.full_like(depth, cam_id, dtype=torch.int32))
            Us.append(u)
            Vs.append(v)

            color = colors[cam_id]
            color = (
                torch.rand(len(depth), 3, device=self._device)
                if color is None
                else color[valid_mask]
            )
            valid_colors.append(color)

        depths = torch.cat(depths, dim=0).reshape(-1, 1)
        indices = torch.cat(indices, dim=0)
        valid_colors = torch.cat(valid_colors, dim=0)
        Us = torch.cat(Us, dim=0)
        Vs = torch.cat(Vs, dim=0)

        points = _depth2camera_points(
            depths, Us, Vs, self._fx, self._fy, self._cx, self._cy, indices
        )
        points = _camera_points2world_points(points, self._poses, indices)

        quats = generate_random_quats(len(points), self._device)
        scales = torch.log(torch.sqrt(distCUDA2(points))).reshape(-1, 1).repeat(1, 3)
        opacities = torch.logit(torch.ones_like(depths) * 0.1)
        features_dc = RGB2SH(valid_colors[..., None, :])
        features_rest = torch.zeros(
            [len(points), self._sh_rest_dim, 3],
            dtype=torch.float32,
            device=self._device,
        )
        uvs = torch.stack([Us, Vs], dim=1)
        uvs = uvs / self._image_size[indices]
        uvs = torch.logit(uvs)

        params = {
            f"uvs{self._suffix}": uvs,
            f"depths{self._suffix}": depths,
            f"quats{self._suffix}": quats,
            f"scales{self._suffix}": scales,
            f"opacities{self._suffix}": opacities,
            f"features_dc{self._suffix}": features_dc,
            f"features_rest{self._suffix}": features_rest,
        }
        self._camera_indices = indices

        if optimizer is None:
            for name, value in params.items():
                self._params_dict[name] = Parameter(
                    value,
                    requires_grad=self._params_dict[name].requires_grad,
                )
        else:
            self.reset_parameters(optimizer, params)

    def prune_parameters(self, optimizer: Optimizer, prune_mask: Tensor) -> None:
        super().prune_parameters(optimizer, prune_mask)
        self._camera_indices = self._camera_indices[~prune_mask]

    def add_parameters(
        self,
        optimizer: Optimizer,
        additional_params: Dict[str, Tensor | Parameter],
        additional_indices: Tensor,
    ) -> None:
        assert len(additional_indices) == len(additional_params["depths"])
        self._camera_indices = torch.cat(
            [self._camera_indices, additional_indices], dim=0
        )
        super().add_parameters(optimizer, additional_params)

    def reset_parameters(
        self,
        optimizer: Optimizer,
        new_params: Dict[str, Tensor | Parameter],
        new_indices: Tensor | None = None,
    ) -> None:
        if new_indices is not None:
            assert len(new_indices) == len(new_params["depths"])
            unique_indices = torch.unique(new_indices)
            if unique_indices.min() < 0 or unique_indices.max() >= self.num_cameras:
                raise ValueError("Invalid camera indices.")
            self._camera_indices = new_indices
        super().reset_parameters(optimizer, new_params)
