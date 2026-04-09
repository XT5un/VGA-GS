from dataclasses import dataclass
from typing import Literal

import gin

try:
    from base_trainer import BaseConfigs
except:
    from train.base_trainer.configs import BaseConfigs


@gin.configurable
@dataclass
class OriginConfigs(BaseConfigs):

    lr_means: float = 0.00016
    """ Learning rate for means optimization. """
    lr_finial: float = 0.0000016
    """ Learning rate for means optimization. """
    lr_delay_mult: float = 0.01
    """ lr means delay multiplier. """
    lr_delay_max_step: int = 30_000
    """ lr means delay max step. """

    lr_scales: float = 0.005
    """ Learning rate for scale optimization. """
    lr_quats: float = 0.001
    """ Learning rate for quaternion optimization. """
    lr_opacities: float = 0.05
    """ Learning rate for opacity optimization. """
    lr_features: float = 0.0025
    """ Learning rate for feature optimization. """

    loss_ssim_lambda: float = 0.2
    """ Weight for SSIM loss. """

    refine_from_step: int = 500
    """ Densify from step. """
    refine_until_step: int = 10_000
    """ Densify until step. """
    densify_interval: int = 500
    """ Densify interval. """
    densify_grad_threshold: float = 0.0002
    """ Densify gradient threshold. """
    percent_dense: float = 0.01
    """ Percent of scene points to be dense. """

    reset_opacity_interval: int = 3000
    """ Opacity reset interval. """
    reset_opacity_threshold: float = 0.01
    """ Opacity reset threshold. """
