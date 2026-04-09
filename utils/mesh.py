import os

import numpy as np
import open3d as o3d
from torch import Tensor

from scene.data import BaseCamera

from .geometry import clean_depthmap

# ! Reference: https://github.com/hbb1/2d-gaussian-splatting/blob/main/render.py
# ! Reference: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/mesh_utils.py


class TSDFMeshExtrator:
    def __init__(
        self,
        depth_trunc: float = 4.0,
        voxel_length: float | None = None,
        sdf_trunc: float | None = None,
        mesh_res: int = 1024,
        clean_depth: bool = False,
    ) -> None:
        voxel_length = depth_trunc / mesh_res if voxel_length is None else voxel_length
        sdf_trunc = 5.0 * voxel_length if sdf_trunc is None else sdf_trunc

        self._depth_trunc = depth_trunc
        self._voxel_length = voxel_length
        self._sdf_trunc = 5.0 * voxel_length
        self._clean_depth = clean_depth

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self._voxel_length,
            sdf_trunc=self._sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def add_frame(
        self,
        rgb: Tensor,
        depth: Tensor,
        camera: BaseCamera,
    ) -> None:
        rgb = np.asarray((rgb.cpu().numpy() * 255), dtype=np.uint8, order="C")
        depth = np.asarray(depth.cpu().numpy(), dtype=np.float32, order="C")
        if self._clean_depth:
            depth, _ = clean_depthmap(depth, camera.model.K.cpu().numpy())

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=self._depth_trunc,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera.model.width,
            camera.model.height,
            camera.model.fx,
            camera.model.fy,
            camera.model.cx,
            camera.model.cy,
        )
        extrinsic = camera.w2c.cpu().numpy().astype(np.float64)
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def extract_and_save(self, path: str) -> None:
        base_dir = os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)

        mesh = self._volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(path, mesh)
