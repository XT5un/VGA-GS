import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Any, Dict, List, Set, Tuple

from absl import app
from base_trainer import BaseTrainer
from configs import OriginConfigs
from rich import print
from strategy import OriginRefinementStrategy

from scene.data import get_scene_radius
from scene.gaussians import Gaussians3D
from scene.losses import *
from utils import (
    colorize,
    read_points,
    write_correspondence,
    write_gray,
    write_points,
    write_rgb,
)


class OriginTrainer(BaseTrainer):

    def __init__(self) -> None:
        super().__init__()

    def _init_configs(self) -> None:
        self._cfg = OriginConfigs()

    def _init_models(self) -> None:
        self._gaussians = Gaussians3D(3, self._device)

    def _init_extra_data(self) -> None:
        super()._init_extra_data()

    def _set_optimizer(self) -> None:
        params_groups = []
        for params in self._gaussians.parameters():
            if params["name"] == "means":
                params["lr"] = self._cfg.lr_means
            elif params["name"] == "scales":
                params["lr"] = self._cfg.lr_scales
            elif params["name"] == "quats":
                params["lr"] = self._cfg.lr_quats
            elif params["name"] == "opacities":
                params["lr"] = self._cfg.lr_opacities
            elif params["name"] == "features_dc":
                params["lr"] = self._cfg.lr_features
            elif params["name"] == "features_rest":
                params["lr"] = self._cfg.lr_features / 20
            else:
                raise ValueError(f"Unknown parameter: {params['name']}")
            params_groups.append(params)
        self._optimizer = torch.optim.Adam(params_groups)

    # def _set_scheduler(self) -> None:

    #     def _get_exp_decay_fn(
    #         lr_init: float, lr_final: float, delay_mult: float, delay_max_step: int
    #     ):
    #         def _exp_decay_fn(step: int):
    #             delay_step = min(step, delay_max_step)
    #             return lr_final + (lr_init - lr_final) * np.exp(
    #                 -delay_mult * delay_step / delay_max_step
    #             )

    #         return _exp_decay_fn

    #     exp_decay_fn = _get_exp_decay_fn(
    #         self._cfg.lr_means,
    #         self._cfg.lr_means_finial,
    #         self._cfg.lr_means_delay_mult,
    #         self._cfg.lr_means_delay_max_step,
    #     )

    #     self._scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         self._optimizer, exp_decay_fn
    #     )

    def _set_strategy(self) -> None:
        self._strategy = OriginRefinementStrategy(
            self._cfg,
            self._step,
            self._optimizer,
            self._gaussians,
            get_scene_radius(self._train_set),
        )

    def _before_train_loop(self) -> None:
        super()._before_train_loop()
        self._gaussians.init_from_points(
            self._init_points,
            self._init_colors,
            optimizer=self._optimizer,
        )

    def _train_step(self) -> Dict[str, Any]:
        end_points = {}

        cam_id = random.randint(0, len(self._train_set) - 1)
        camera = self._train_set[cam_id]

        render_rlt = self._renderer(
            self._gaussians.get_render_parameters(self._degree_to_use),
            camera.model,
            camera.w2c,
            bg_color=self._bg_color,
            need_extra_infos=True,
        )
        end_points["radii"] = render_rlt.extra["radii"]
        end_points["means2d"] = render_rlt.extra["means2d"]

        loss_l1 = compute_l1_loss(render_rlt.image, camera.image)
        loss_l1 *= 1 - self._cfg.loss_ssim_lambda
        end_points["loss_l1"] = loss_l1

        loss_ssim = compute_ssim_loss(render_rlt.image, camera.image)
        loss_ssim *= self._cfg.loss_ssim_lambda
        end_points["loss_ssim"] = loss_ssim

        loss = loss_l1 + loss_ssim
        end_points["loss"] = loss

        return end_points


def main(argv) -> None:
    trainer = OriginTrainer()
    trainer.train()


if __name__ == "__main__":
    app.run(main)
