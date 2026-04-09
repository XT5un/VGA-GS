from dataclasses import dataclass, field
from typing import Any, List, Literal

import gin
from absl import flags

from scene.data import SUPPORTED_DATASETS

flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings."
)


def parse_gin_configs():
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_file, flags.FLAGS.gin_param)


def dump_gin_configs(output_dir: str):
    with open(f"{output_dir}/configs.gin", "w") as f:
        f.write(gin.operative_config_str())


def load_gin_configs(gin_file: str):
    with open(gin_file, "r") as f:
        gin_str = f.read()
    gin.parse_config(gin_str)


@gin.configurable
@dataclass
class BaseConfigs:

    rasterizer_backend: Literal["RaDe", "2dgs", "3dgs"] = "RaDe"
    """ backend of the rasterizer """

    # Data configs
    dataset_name: SUPPORTED_DATASETS = gin.REQUIRED
    """ name of the dataset """
    n_sparse_views: int = -1
    """ number of sparse views."""
    with_background: bool = False
    """ background need to be special treated """

    """ name of the dataset """
    data_root_dir: str = gin.REQUIRED
    """ root directory of the dataset """
    scene_name: str = gin.REQUIRED
    """ name of the scene """
    init_points_file: str = ""
    """ path to the initial point cloud """
    resize_factor: int = 1
    """ resize factor of the images """
    data_device: str = "cuda"
    """ device for storing the data """

    # Training configs
    max_steps: int = 30_000
    """ maximum number of training steps """
    print_interval: int = 100
    """ interval of printing training information """
    eval_steps: List[int] = field(default_factory=list)
    """ steps to evaluate the model """
    save_steps: List[int] = field(default_factory=list)
    """ steps to save the model """
    output_root_dir: str = "outs"
    """ root directory of the output """
    exp_name: str = ""
    """ name of the experiment """

    # Gaussian configs
    sh_degree: int = 3
    """ degree of spherical harmonics """
    bg_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """ background color of the images """

    def __post_init__(self):
        # set the default values
        if len(self.eval_steps) == 0:
            self.eval_steps = [self.max_steps // 4, self.max_steps]
        if len(self.save_steps) == 0:
            self.save_steps = [self.max_steps // 4, self.max_steps]
        if self.exp_name == "":
            self.exp_name = f"debug_{self.rasterizer_backend}"
