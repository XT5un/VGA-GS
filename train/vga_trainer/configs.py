from dataclasses import dataclass, field
from typing import List, Literal

import gin

try:
    from origin_trainer import OriginConfigs
except:
    from train.origin_trainer.configs import OriginConfigs

from scene.data import SUPPORTED_DATASETS
from scene.priors_manager import SUPPORTED_DEPTH_METHODS


@gin.configurable
@dataclass
class VGAGSConfigs(OriginConfigs):

    depth_estimation_method: SUPPORTED_DEPTH_METHODS = "Metric3D"
    """ Method for depth estimation. """
    image_match_method: Literal["DKM"] = "DKM"
    """ Method for matching images. """
    use_cache_data: bool = True
    """ Use cache data. """
    scene_type: Literal["indoor", "outdoor"] = "indoor"
    """ Scene type. Only for DepthAnything model. """

    splatter_type: Literal["3d", "ray"] = "ray"
    """ Type of gaussian splatter. """
    sample_important_pixels: bool = True
    """ Sample important pixels. If false, use all pixels for initialization. """
    patch_size_for_sample: int = 15
    """ Patch size for sampling. """
    max_num_splatter_per_view: int = -1
    """ Maximum number of initial splatter per view. """

    prune_opacity_from: int = 3000
    """ Prune opacity from this step. """
    prune_opacity_until: int = int(1e8)
    """ Prune opacity until this step. """
    prune_opacity_interval: int = 1000
    """ Prune opacity interval. """
    prune_opacity_threshold: float = 0.1
    """ Prune gaussian if opacity is smaller than this threshold. """

    reset_scale_until: int = -1
    """ Reset scale until this step. -1 means no reset. """
    reset_scale_interval: int = 50
    """ Reset scale interval. """

    densify_from_view_at: List[int] = field(default_factory=lambda: [2000, 9000])
    """ Densify from view as this step. """
    downsample_voxel_res: int = -1
    """ Downsample voxel resolution when densification. """

    lr_depths: float = 1e-4
    """ Learning rate for gaussian depths. """
    lr_uvs: float = 1e-4
    """ Learning rate for gaussian uvs. """

    fusion_depth_ratio: float = 0.1
    """ Ratio for fusion mean depth and midian depth. 0 means only use mean depth. """

    loss_group_depth_rank_level: int = 128
    """ Number of levels for group depth rank loss. """
    loss_group_depth_rank_lambda: float = 1.0
    """ Weight for group depth rank loss. -1 means no group depth rank loss. """
    loss_normal_lambda: float = 0.05
    """ Weight for normal loss. -1 means no normal loss. """
    use_mono_depth_normal_consistency: bool = True
    """ Use mono depth to supervise normal. If false, use render depth. """
    loss_depth_close_threshold: float = 0.1
    """ Threshold for close depth loss. Only for GaussiansRay. """
    loss_depth_close_lambda: float = 0.01
    """ Weight for close depth loss. Only for GaussiansRay. """

    train_pseudo_view_from: int = 7000
    """ Train pseudo view from this step. """
    train_pseudo_view_until: int = int(1e8)
    """ Train pseudo view until this step. -1 means no pseudo view training. """
    train_pseudo_view_interval: int = 5
    """ Train pseudo view interval. """
    loss_pseudo_view_weight: float = 0.2
    """ Weight for pseudo view loss. """
    warp_midian_depth: bool = False
    """ Use midian depth for frame warping. """
    pseudo_trans_range: float = 3.0
    """ Translation differece between pseudo and train. """
    pseudo_angle_range: float = 30.0
    """ Angle difference between pseudo and train. """

    def __post_init__(self):
        if self.exp_name == "":
            self.exp_name = f"debug_{self.splatter_type}_{self.depth_estimation_method}_{self.rasterizer_backend}"
        if len(self.eval_steps) == 0:
            self.eval_steps = sorted(list(set([self.max_steps])))
        if len(self.save_steps) == 0:
            self.save_steps = sorted(list(set([self.max_steps])))
