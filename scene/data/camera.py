import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pypose import euler2SO3
from torch import Tensor

__all__ = [
    "PinholeModel",
    "points2pixels",
    "depth2points",
    "points2normal",
    "depth2normal",
    "transform_points",
    "BaseCamera",
    "DataCamera",
    "warp_from_depth",
]


class PinholeModel:
    """
    This class is used to store the camera intrinsic parameters.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        device: torch.device,
    ) -> None:
        self._width = width
        self._height = height
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._device = device
        self._K = torch.tensor(
            [
                [self._fx, 0.0, self._cx],
                [0.0, self._fy, self._cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self._device,
        )
        self._fovx = 2 * math.atan(self._width / (2 * self._fx))
        self._fovy = 2 * math.atan(self._height / (2 * self._fy))
        self._projection_matrix = self._get_opengl_project_matrix(0.01, 100.0)

    @property
    def K(self) -> Tensor:
        """opencv style camera matrix, shape (3, 3)"""
        return self._K

    @property
    def projection_matrix(self) -> Tensor:
        """OpenGL style projection matrix, shape (4, 4)"""
        return self._projection_matrix

    @property
    def FoVx(self) -> float:
        return self._fovx

    @property
    def FoVy(self) -> float:
        return self._fovy

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fx(self) -> float:
        return self._fx

    @property
    def fy(self) -> float:
        return self._fy

    @property
    def cx(self) -> float:
        return self._cx

    @property
    def cy(self) -> float:
        return self._cy

    @property
    def device(self) -> torch.device:
        return self._device

    def _get_opengl_project_matrix(self, znear: float, zfar) -> Tensor:
        t = znear * math.tan(0.5 * self._fovy)
        b = -t
        r = znear * math.tan(0.5 * self._fovx)
        l = -r
        n = znear
        f = zfar

        P = torch.tensor(
            [
                [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
                [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
                [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=self._device,
        )

        return P


def points2pixels(
    points: Tensor, camera_model: PinholeModel, w2c: Tensor | None = None
) -> Tuple[Tensor, Tensor, Tensor]:
    # world to camera
    if w2c is not None:
        points = transform_points(points, w2c)

    # camera to image
    prjs = project_points(points, camera_model.K)

    # normalize
    depth = prjs[..., 2]
    pixels = prjs[..., :2] / depth[..., None]

    # check if the points are visible
    visible_mask = (
        (depth > 0.01)
        & (pixels[..., 0] >= 0)
        & (pixels[..., 0] <= camera_model.width - 1)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] <= camera_model.height - 1)
    )

    return pixels, depth, visible_mask


def depth2points(
    depth: Tensor,
    camera_model: PinholeModel,
    pixels: Tensor | None = None,
    c2w: Tensor | None = None,
) -> Tensor:
    if pixels is not None:
        assert (
            depth.shape == pixels.shape[:-1]
        ), "depth and pixels should have the same shape"
        assert (
            depth.dtype == pixels.dtype
        ), "depth and pixels should have the same dtype"
    else:
        H, W = depth.shape
        assert H == camera_model.height and W == camera_model.width, "invalid shape"
        pixels = torch.meshgrid(
            torch.arange(W, device=depth.device, dtype=depth.dtype) + 0.5,
            torch.arange(H, device=depth.device, dtype=depth.dtype) + 0.5,
            indexing="xy",
        )
        pixels = torch.stack(pixels, dim=-1)

    # pixels to camera space
    points = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    points *= depth[..., None]
    K_inv = torch.inverse(camera_model.K)
    points = project_points(points, K_inv)

    # camera space to world space
    if c2w is not None:
        points = transform_points(points, c2w)

    return points


def points2normal(points: Tensor) -> Tensor:
    """
    calculate normal from points in camera space.

    Args:
        points: (H, W, 3)
    Returns:
        normal: (H, W, 3)
    """
    assert points.shape[-1] == 3, "last dimension should be 3"
    assert points.ndim == 3, "points should have 3 dimensions"
    dx = points[2:, 1:-1, :] - points[:-2, 1:-1, :]  # (H-2, W-2, 3)
    dy = points[1:-1, 2:, :] - points[1:-1, :-2, :]  # (H-2, W-2, 3)
    normal = torch.cross(dx, dy, dim=-1)  # (H-2, W-2, 3)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    normal = torch.nan_to_num(normal, 0.0, 0.0)

    normal = torch.cat(
        [normal[:1, :, :], normal, normal[-1:, :, :]], dim=0
    )  # (H, W-2, 3)
    normal = torch.cat(
        [normal[:, :1, :], normal, normal[:, -1:, :]], dim=1
    )  # (H, W, 3)
    return normal


def depth2normal(
    depth: Tensor, camera_model: PinholeModel, pose: Tensor | None = None
) -> Tensor:
    points = depth2points(depth, camera_model)
    if pose is not None:
        points = transform_points(points, pose)

    return points2normal(points)


def transform_points(points: Tensor, trans_mat: Tensor) -> Tensor:
    assert points.shape[-1] == 3, "last dimension should be 3"
    assert trans_mat.shape == (4, 4), "pose should have shape (4, 4)"

    points_H = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_H = torch.einsum("ij,...j->...i", trans_mat, points_H)
    return points_H[..., :3]


def project_points(points: Tensor, prj_mat: Tensor) -> Tensor:
    assert points.shape[-1] == 3, "last dimension should be 3"
    assert prj_mat.shape == (3, 3), "K should have shape (3, 3)"

    points = torch.einsum("ij,...j->...i", prj_mat, points)
    return points


class BaseCamera:
    """
    This class is only used to process the camera projection.
    It is just a wrapper of virtual camera.
    """

    def __init__(
        self,
        cam2world: Tensor,
        camera_model: PinholeModel,
        device: torch.device,
    ) -> None:
        self._device = device
        self._c2w = cam2world.to(self._device)
        self._model = camera_model

    @property
    def c2w(self) -> Tensor:
        return self._c2w

    @c2w.setter
    def c2w(self, value: Tensor) -> None:
        self._c2w = value.to(self._device)

    @property
    def w2c(self) -> Tensor:
        return torch.inverse(self._c2w)

    @w2c.setter
    def w2c(self, value: Tensor) -> None:
        self._c2w = torch.inverse(value.to(self._device))

    @property
    def model(self) -> PinholeModel:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def points2pixels(
        self, points: Tensor, in_world: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if in_world:
            w2c = self.w2c
        else:
            w2c = None

        return points2pixels(points, self.model, w2c)

    def depth2points(
        self, depth: Tensor, pixels: Tensor | None = None, to_world: bool = True
    ) -> Tensor:
        return depth2points(depth, self.model, pixels, self.c2w if to_world else None)

    def depth2normal(self, depth: Tensor, to_world=False) -> Tensor:
        points = self.depth2points(depth, to_world=to_world)
        return points2normal(points)

    def create_pseudo_camera(
        self,
        trans_range: float = 0.1,
        angle_range: float = 10.0,
        camera_model: PinholeModel | None = None,
        use_rad: bool = False,
    ) -> "BaseCamera":
        if camera_model is None:
            camera_model = self.model

        trans_diff = (torch.rand(3, device=self.device) - 0.5) * trans_range
        angle_diff = (torch.rand(3, device=self.device) - 0.5) * angle_range
        if not use_rad:
            angle_diff = torch.deg2rad(angle_diff)

        rotmat = euler2SO3(angle_diff).matrix()
        pose_diff = torch.eye(4, device=self.device)
        pose_diff[:3, :3] = rotmat
        pose_diff[:3, 3] = trans_diff  # pseudo view to this view

        p_c2w = self.c2w @ pose_diff

        return BaseCamera(p_c2w, camera_model, self.device)

    def to_dict(self) -> Dict[str, Any]:
        meta = {
            "width": self.model.width,
            "height": self.model.height,
            "intrinsic": self.model.K.cpu().tolist(),
            "c2w": self.c2w.cpu().tolist(),
        }
        return meta


class DataCamera(BaseCamera):
    """This class is used to store the real data."""

    def __init__(
        self,
        cam2world: Tensor,
        camera_model: PinholeModel,
        device: torch.device,
        data_device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(cam2world, camera_model, device)

        self._data_device = data_device if data_device is not None else device
        self._name = name

        self._image: Tensor | None = None
        self._image_mask: Tensor | None = None

        self._depth: Tensor | None = None
        self._depth_mask: Tensor | None = None

    @property
    def image(self) -> Tensor | None:
        if self._image is None:
            return None
        return self._image.to(self._device)

    @image.setter
    def image(self, value: Tensor) -> None:
        self._image = value.to(self._data_device)

    @property
    def image_mask(self) -> Tensor | None:
        if self._image_mask is None:
            return None
        return self._image_mask.to(self._device)

    @image_mask.setter
    def image_mask(self, value: Tensor) -> None:
        self._image_mask = value.to(self._data_device)

    @property
    def depth(self) -> Tensor | None:
        if self._depth is None:
            return None
        return self._depth.to(self._device)

    @depth.setter
    def depth(self, value: Tensor) -> None:
        self._depth = value.to(self._data_device)

    @property
    def depth_mask(self) -> Tensor | None:
        if self._depth_mask is None:
            return None
        return self._depth_mask.to(self._device)

    @depth_mask.setter
    def depth_mask(self, value: Tensor) -> None:
        self._depth_mask = value.to(self._data_device)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def data_device(self) -> torch.device:
        return self._data_device

    def create_pseudo_camera(
        self,
        trans_range: float = 0.1,
        angle_range: float = 10,
        camera_model: PinholeModel | None = None,
        use_rad: bool = False,
    ) -> "DataCamera":
        pseudo_base_cam = super().create_pseudo_camera(
            trans_range, angle_range, camera_model, use_rad
        )

        return DataCamera(
            pseudo_base_cam.c2w,
            pseudo_base_cam.model,
            pseudo_base_cam.device,
            self._data_device,
            self._name,
        )


def warp_from_depth(
    feats_src: Tensor,
    depth_dst: Tensor,
    camera_src: BaseCamera,
    camera_dst: BaseCamera,
) -> Tensor:
    points_dst = camera_dst.depth2points(depth_dst, to_world=True)
    prj_pixels, prj_depth, vis_mask = camera_src.points2pixels(
        points_dst, in_world=True
    )

    # map pixels to [-1, +1]
    width = camera_src.model.width
    height = camera_src.model.height
    prj_pixels[..., 0] = (prj_pixels[..., 0] / width) * 2 - 1
    prj_pixels[..., 1] = (prj_pixels[..., 1] / height) * 2 - 1

    feats_src = feats_src[None, ...].permute(0, 3, 1, 2)
    prj_pixels = prj_pixels[None, ...]
    feats_warped = torch.nn.functional.grid_sample(
        feats_src, prj_pixels, align_corners=True
    )
    feats_warped = feats_warped[0].permute(1, 2, 0)

    return feats_warped
