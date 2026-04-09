import math
from typing import Any, Callable, Dict

import torch
from diff_gaussian_rasterization_2dgs import (
    GaussianRasterizationSettings as RasterizationSettings_2DGS,
)
from diff_gaussian_rasterization_2dgs import GaussianRasterizer as Rasterizer_2DGS
from diff_gaussian_rasterization_RaDe import (
    GaussianRasterizationSettings as RasterizationSettings_RaDe,
)
from diff_gaussian_rasterization_RaDe import GaussianRasterizer as Rasterizer_RaDe
from torch import Tensor

from .data import PinholeModel, depth2points
from .gaussians import RenderGaussiansParams
from .gaussians.utils import quaternions_to_matrixes

__all__ = [
    "renderer_RaDe",
    "renderer_2DGS",
    "RenderResult",
    "RendererFunction",
]


class RenderResult:
    def __init__(
        self,
        image: Tensor,
        alpha: Tensor,
        depth: Tensor,
        normal: Tensor | None = None,
        mid_depth: Tensor | None = None,
        pointmap: Tensor | None = None,
        mid_pointmap: Tensor | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        self.image = image
        self.alpha = alpha
        self.depth = depth
        self.normal = normal
        self.mid_depth = mid_depth
        self.pointmap = pointmap
        self.mid_pointmap = mid_pointmap
        self.extra = extra


RendererFunction = Callable[
    [RenderGaussiansParams, PinholeModel, Tensor, Tensor | None, bool, bool],
    RenderResult,
]


def renderer_RaDe(
    render_parameters: RenderGaussiansParams,
    camera_model: PinholeModel,
    world2camera: Tensor,
    bg_color: Tensor | None = None,
    need_extra_infos: bool = False,
    require_coord: bool = False,
) -> RenderResult:
    means2D = torch.zeros_like(render_parameters.means, requires_grad=True)

    tanfovx = math.tan(camera_model.FoVx * 0.5)
    tanfovy = math.tan(camera_model.FoVy * 0.5)
    world_view_transform = world2camera.transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(
            camera_model.projection_matrix.transpose(0, 1).unsqueeze(0)
        )
    ).squeeze(0)
    camera_position = torch.inverse(world2camera)[:3, 3]

    if bg_color is None:
        # default background color is black
        bg_color = torch.zeros(3, dtype=means2D.dtype, device=means2D.device)
    else:
        bg_color = bg_color.to(means2D.device)

    raster_settings = RasterizationSettings_RaDe(
        image_height=camera_model.height,
        image_width=camera_model.width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=render_parameters.degree_to_use,
        campos=camera_position,
        prefiltered=False,
        debug=False,
        kernel_size=0.0,
        require_depth=True,
        require_coord=require_coord,
    )

    rasterizer = Rasterizer_RaDe(raster_settings=raster_settings)
    colors_precomp = render_parameters.precomputed_colors
    shs = render_parameters.sh_coeffs if colors_precomp is None else None

    (
        rendered_image,
        radii,
        rendered_coord,
        rendered_midcoord,
        rendered_depth,
        rendered_middepth,
        rendered_alpha,
        rendered_normal,
    ) = rasterizer(
        means3D=render_parameters.means,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=render_parameters.opacities,
        scales=render_parameters.scales,
        rotations=render_parameters.quats,
        cov3D_precomp=None,
    )

    rendered_image = torch.clamp(rendered_image, 0.0, 1.0).permute(1, 2, 0)
    rendered_alpha = rendered_alpha.squeeze(0)
    rendered_depth = rendered_depth.squeeze(0)
    rendered_middepth = rendered_middepth.squeeze(0)
    rendered_normal = rendered_normal.permute(1, 2, 0)
    pointmap = None
    mid_pointmap = None
    if require_coord:
        pointmap = rendered_coord.permute(1, 2, 0)
        mid_pointmap = rendered_midcoord.permute(1, 2, 0)

    extra = None
    if need_extra_infos:
        extra = {}
        means2D.retain_grad()
        extra["means2d"] = means2D
        extra["radii"] = radii

    render_rlt = RenderResult(
        image=rendered_image,
        alpha=rendered_alpha,
        depth=rendered_depth,
        normal=rendered_normal,
        mid_depth=rendered_middepth,
        pointmap=pointmap,
        mid_pointmap=mid_pointmap,
        extra=extra,
    )

    return render_rlt


def renderer_2DGS(
    render_parameters: RenderGaussiansParams,
    camera_model: PinholeModel,
    world2camera: Tensor,
    bg_color: Tensor | None = None,
    need_extra_infos: bool = False,
    require_coord: bool = False,
):
    means2D = torch.zeros_like(render_parameters.means, requires_grad=True)

    tanfovx = math.tan(camera_model.FoVx * 0.5)
    tanfovy = math.tan(camera_model.FoVy * 0.5)
    world_view_transform = world2camera.transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(
            camera_model.projection_matrix.transpose(0, 1).unsqueeze(0)
        )
    ).squeeze(0)
    camera_position = torch.inverse(world2camera)[:3, 3]

    if bg_color is None:
        # default background color is black
        bg_color = torch.zeros(3, dtype=means2D.dtype, device=means2D.device)
    else:
        bg_color = bg_color.to(means2D.device)

    raster_settings = RasterizationSettings_2DGS(
        image_height=camera_model.height,
        image_width=camera_model.width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=render_parameters.degree_to_use,
        campos=camera_position,
        prefiltered=False,
        debug=False,
    )

    rasterizer = Rasterizer_2DGS(raster_settings=raster_settings)
    colors_precomp = render_parameters.precomputed_colors
    shs = render_parameters.sh_coeffs if colors_precomp is None else None
    scales = render_parameters.scales[..., :2]  # only has 2-dim scales

    rendered_image, radii, allmap = rasterizer(
        means3D=render_parameters.means,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=render_parameters.opacities,
        scales=scales,
        rotations=render_parameters.quats,
        cov3D_precomp=None,
    )

    rendered_image = torch.clamp(rendered_image, 0.0, 1.0).permute(1, 2, 0)
    rendered_alpha = allmap[1:2].squeeze(0)
    rendered_depth = allmap[0:1].squeeze(0)
    rendered_depth = rendered_depth / rendered_alpha
    rendered_depth = torch.nan_to_num(rendered_depth, 0.0, 0.0)
    rendered_middepth = allmap[5:6].squeeze(0)
    rendered_middepth = torch.nan_to_num(rendered_middepth, 0.0, 0.0)
    rendered_normal = allmap[2:5].permute(1, 2, 0)

    extra = None
    if need_extra_infos:
        extra = {}
        means2D.retain_grad()
        extra["means2d"] = means2D
        extra["radii"] = radii

    if require_coord:
        pointmap = depth2points(rendered_depth, camera_model)
        mid_pointmap = depth2points(rendered_middepth, camera_model)
    else:
        pointmap = None
        mid_pointmap = None

    render_rlt = RenderResult(
        image=rendered_image,
        alpha=rendered_alpha,
        depth=rendered_depth,
        normal=rendered_normal,
        mid_depth=rendered_middepth,
        pointmap=pointmap,
        mid_pointmap=mid_pointmap,
        extra=extra,
    )

    return render_rlt
