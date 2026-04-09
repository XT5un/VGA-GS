import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Type

import numpy as np
import torch
from rich import print
from rich.align import Align
from rich.progress import track
from rich.table import Table

from scene.data import GroundTruthDataset
from scene.gaussians import BaseGaussians
from scene.metrics import MultiMetrics
from scene.renderer import *
from utils import colorize, read_points, write_gray, write_rgb, write_rgba

from .configs import BaseConfigs, dump_gin_configs, parse_gin_configs
from .strategy import BaseRefinementStrategy


class BaseTrainer(ABC):
    def __init__(self) -> None:
        parse_gin_configs()

        # initialize cfgs, datasets, models
        self._init_configs()
        self._device = torch.device("cuda")
        self._data_device = torch.device(self._cfg.data_device)
        self._bg_color = torch.tensor(
            self._cfg.bg_color, device=self._device, dtype=torch.float32
        )
        self._init_datasets()
        self._init_extra_data()
        self._init_models()

        # setting the rasterizer
        self._renderer = renderer_RaDe
        if self._cfg.rasterizer_backend == "RaDe":
            self._renderer = renderer_RaDe
        elif self._cfg.rasterizer_backend == "2dgs":
            self._renderer = renderer_2DGS
        elif self._cfg.rasterizer_backend == "3dgs":
            raise NotImplementedError("3DGS rasterizer is not implemented.")
        else:
            raise ValueError(f"Unknown rasterizer: {self._cfg.rasterizer_backend}")

        # initialize some variables
        self._step: int = 0
        self._begin_step: int = 0
        self._metrics = MultiMetrics(self._device)

        if self._cfg.n_sparse_views < 0:
            self._save_dir = os.path.join(
                self._cfg.output_root_dir,
                self._cfg.exp_name,
                self._cfg.dataset_name,
                self._cfg.scene_name,
            )
        else:
            self._save_dir = os.path.join(
                self._cfg.output_root_dir,
                self._cfg.exp_name,
                f"{self._cfg.dataset_name}-{self._cfg.n_sparse_views}_views",
                self._cfg.scene_name,
            )

        self._model_dir = os.path.join(self._save_dir, "models")
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)
        self._loss_records: Dict[str, float] = {}
        self._metrics_records: Dict[
            Literal["train", "test"],
            List[Dict[Literal["step", "psnr", "ssim", "lpips", "rmse"], float]],
        ] = {}

        # call _after_init to do some operations after initialization
        self._after_init()

    @property
    def _degree_to_use(self) -> int:
        """Degree of SH to use when this step."""
        return min(self._step // 1000, self._cfg.sh_degree)

    def _init_configs(self) -> None:
        """Initialize configurations."""
        self._cfg = BaseConfigs()

    def _init_datasets(self) -> None:
        """Initialize datasets."""
        self._train_set = GroundTruthDataset(
            self._cfg.dataset_name,
            self._cfg.data_root_dir,
            self._cfg.scene_name,
            self._cfg.resize_factor,
            self._cfg.n_sparse_views,
            mode="train",
            device=self._device,
            data_device=self._data_device,
        )
        self._test_set = GroundTruthDataset(
            self._cfg.dataset_name,
            self._cfg.data_root_dir,
            self._cfg.scene_name,
            self._cfg.resize_factor,
            self._cfg.n_sparse_views,
            mode="test",
            device=self._device,
            data_device=self._data_device,
        )

    def _init_models(self) -> None:
        """Initialize models."""
        self._gaussians = BaseGaussians(self._cfg.sh_degree, self._device)

    def _init_extra_data(self) -> None:
        """Initialize extra data."""
        self._init_points = torch.empty(0, 3, dtype=torch.float32, device=self._device)
        self._init_colors = torch.empty(0, 3, dtype=torch.float32, device=self._device)

        if self._cfg.init_points_file == "":
            warnings.warn("There is a initial points file")
            return

        points_path = os.path.join(
            self._cfg.data_root_dir, self._cfg.scene_name, self._cfg.init_points_file
        )
        if os.path.exists(points_path):
            points, colors = read_points(points_path)
            self._init_points = torch.from_numpy(points).to(self._device)
            self._init_colors = torch.from_numpy(colors).to(self._device)
        else:
            warnings.warn(
                f"Initial points file not found: {points_path}, use random points.",
                UserWarning,
            )
            self._init_points = torch.randn(
                1000, 3, dtype=torch.float32, device=self._device
            )
            self._init_colors = torch.rand(
                1000, 3, dtype=torch.float32, device=self._device
            )

        return

    def _after_init(self) -> None:
        """Operations after all initialization."""
        # save configurations
        dump_gin_configs(self._save_dir)
        # print configurations and dataset information
        print(self._cfg)
        print(f"Train set: {len(self._train_set)} samples.")
        print(f"Test set: {len(self._test_set)} samples.")
        return

    @abstractmethod
    def _set_optimizer(self) -> None:
        """Set optimizer for training."""
        self._optimizer = torch.optim.Optimizer()
        raise NotImplementedError("Method `_set_optimizer` must be implemented.")

    def _set_scheduler(self) -> None:
        """Set scheduler for training."""
        self._scheduler = None

    def _set_strategy(self) -> None:
        """Set strategy for training."""
        self._strategy = BaseRefinementStrategy(
            self._cfg, self._begin_step, self._optimizer, self._gaussians
        )

    def _restore_model(self) -> bool:
        """Restore model from checkpoint."""
        # TODO: 检查是否存在ckpt 如果有就恢复模型
        return False

    def _before_train_step(self) -> None:
        """
        Operations before each training step.

        It will be called before `_train_step`.
        """
        return

    def _after_train_step(self) -> None:
        """
        Operations after each training step.

        It will be called after `_train_step` and `self._optimizer.step`.
        """
        return

    @abstractmethod
    def _train_step(self) -> Dict[str, Any]:
        """Training step. Return the loss and other information."""
        raise NotImplementedError("Method `_train_step` must be implemented.")

    def _record_train_info(
        self, end_points: Dict[str, Any], step: int | None = None
    ) -> None:
        """Record training information."""
        step = step if step is not None else self._step

        for k, v in end_points.items():
            if k == "loss" or k.startswith("loss_"):
                if torch.is_tensor(v):
                    v = v.item()
                self._loss_records.setdefault(k, []).append(v)

        if (step + 1) % self._cfg.print_interval == 0:
            mean_loss_records = {k: np.mean(v) for k, v in self._loss_records.items()}
            buffer = f":cat:[red]Step {step + 1:<5d} | :rabbit:[yellow]Loss: {mean_loss_records['loss']:.4f}"
            for k, v in mean_loss_records.items():
                if k != "loss":
                    buffer += f" [cyan]{k}: {v:.4f}"
            print(buffer)
            self._loss_records = {}

    @torch.no_grad()
    def _evaluate(
        self,
        dataset: GroundTruthDataset,
        mode: str | None = None,
        step: int | None = None,
    ) -> None:
        """Evaluate the model on the dataset."""
        step = step if step is not None else self._step
        if mode is None:
            eval_dir = os.path.join(self._save_dir, "evals", f"iter_{step + 1}")
        else:
            eval_dir = os.path.join(self._save_dir, "evals", f"iter_{step + 1}", mode)
        os.makedirs(eval_dir, exist_ok=True)

        all_metrics = {"name": []}
        for camera in dataset:
            render_rlt = self._renderer(
                self._gaussians.get_render_parameters(self._degree_to_use),
                camera.model,
                camera.w2c,
                bg_color=self._bg_color,
                need_extra_infos=False,
                require_coord=True,
            )

            metrics = self._metrics(
                render_rlt.image,
                camera.image,
                render_rlt.depth,
                camera.depth,
                camera.image_mask,
                camera.depth_mask,
            )
            all_metrics["name"].append(camera.name)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

            # write the render results
            valid_mask = render_rlt.alpha > 0.1
            if camera.image_mask is not None:
                write_rgba(
                    os.path.join(eval_dir, f"{camera.name}_image_pred_masked.png"),
                    render_rlt.image.cpu().numpy(),
                    camera.image_mask.float().cpu().numpy(),
                )
            write_rgb(
                os.path.join(eval_dir, f"{camera.name}_image_pred.jpg"),
                render_rlt.image.cpu().numpy(),
            )
            write_rgb(
                os.path.join(eval_dir, f"{camera.name}_image_gt.jpg"),
                camera.image.cpu().numpy(),
            )
            write_gray(
                os.path.join(eval_dir, f"{camera.name}_alpha_pred.jpg"),
                render_rlt.alpha.cpu().numpy(),
            )
            write_rgb(
                os.path.join(eval_dir, f"{camera.name}_depth_pred.jpg"),
                colorize(
                    render_rlt.depth.cpu().numpy(),
                    mask=valid_mask.cpu().numpy(),
                ),
            )
            if render_rlt.mid_depth is not None:
                write_rgb(
                    os.path.join(eval_dir, f"{camera.name}_mid_depth_pred.jpg"),
                    colorize(
                        render_rlt.mid_depth.cpu().numpy(),
                        mask=valid_mask.cpu().numpy(),
                    ),
                )
            if camera.depth is not None:
                write_rgb(
                    os.path.join(eval_dir, f"{camera.name}_depth_gt.jpg"),
                    colorize(
                        camera.depth.cpu().numpy(),
                        mask=camera.depth_mask.cpu().numpy(),
                    ),
                )
            # points = transform_points(render_rlt.pointmap, camera.c2w)
            # write_points(
            #     os.path.join(eval_dir, f"{camera.name}_points_pred.ply"),
            #     points[valid_mask].cpu().numpy(),
            #     render_rlt.image[valid_mask].cpu().numpy(),
            # )
            # if render_rlt.mid_pointmap is not None:
            #     points = transform_points(render_rlt.mid_pointmap, camera.c2w)
            #     write_points(
            #         os.path.join(eval_dir, f"{camera.name}_mid_points_pred.ply"),
            #         points[valid_mask].cpu().numpy(),
            #         render_rlt.image[valid_mask].cpu().numpy(),
            #     )
            if render_rlt.normal is not None:
                write_rgb(
                    os.path.join(eval_dir, f"{camera.name}_normal_pred.jpg"),
                    ((render_rlt.normal + 1) / 2).cpu().numpy(),
                )
        mean_metrics = {}
        for k, v in all_metrics.items():
            if k != "name":
                mean_metrics[k] = np.mean(v).item()

        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            meta = {"frames": [], "mean_metrics": mean_metrics}
            for i, name in enumerate(all_metrics["name"]):
                meta["frames"].append(
                    {
                        "name": name,
                        "metrics": {
                            k: v[i] for k, v in all_metrics.items() if k != "name"
                        },
                    }
                )
            json.dump(meta, f, indent=4)

        if mode in ("train", "test"):
            self._metrics_records.setdefault(mode, []).append(
                {"step": step + 1, **mean_metrics}
            )

    def _print_metrics(self) -> None:
        """Print evaluation results."""
        table = Table(
            title="Evaluation Results",
            show_header=True,
            show_lines=True,
            show_edge=True,
        )
        table.add_column("Dataset", style="green", justify="center")
        table.add_column("Step", style="red", justify="center")
        table.add_column("PSNR", style="blue", justify="center")
        table.add_column("SSIM", style="blue", justify="center")
        table.add_column("LPIPS", style="blue", justify="center")
        table.add_column("RMSE", style="cyan", justify="center")

        for mode, metrics_list in self._metrics_records.items():
            step_table = Table(show_header=False, show_lines=False, show_edge=False)
            psnr_table = Table(show_header=False, show_lines=False, show_edge=False)
            ssim_table = Table(show_header=False, show_lines=False, show_edge=False)
            lpips_table = Table(show_header=False, show_lines=False, show_edge=False)
            rmse_table = Table(show_header=False, show_lines=False, show_edge=False)

            for metrics in metrics_list:
                step_table.add_row(f"{metrics['step']}")
                psnr_table.add_row(f"{metrics['psnr']:.4f}")
                ssim_table.add_row(f"{metrics['ssim']:.4f}")
                lpips_table.add_row(f"{metrics['lpips']:.4f}")
                rmse_table.add_row(f"{metrics['rmse']:.4f}")

            table.add_row(
                Align(mode, vertical="middle", align="center"),
                step_table,
                psnr_table,
                ssim_table,
                lpips_table,
                rmse_table,
            )
        print(table)

    def _before_train_loop(self) -> None:
        """
        Operations before training.

        It will be called before the training loop.
        """
        self._set_optimizer()
        self._set_scheduler()
        self._set_strategy()

        self._step = self._begin_step

    def _after_train_loop(self) -> None:
        """
        Operations after training.

        It will be called after the training loop.
        """
        return

    def train(self) -> None:
        self._before_train_loop()

        usage_time = 0.0
        usage_memory = []
        # training loop
        for step in track(
            range(self._begin_step, self._cfg.max_steps),
            description=":fire:Train:fire:",
        ):
            torch.cuda.synchronize()
            bgn_time = time.perf_counter()

            self._step = step
            self._before_train_step()
            end_points = self._train_step()
            loss: torch.Tensor = end_points["loss"]
            self._optimizer.zero_grad()
            loss.backward()

            for k, v in self._gaussians.params_dict.items():
                if torch.isnan(v.grad).any():
                    torch.nan_to_num_(v.grad, 0.0)

            self._strategy.step(end_points)

            self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
            self._strategy.step_after_backward(end_points)

            torch.cuda.synchronize()
            usage_time += time.perf_counter() - bgn_time

            self._record_train_info(end_points)
            self._after_train_step()

            if (step + 1) in self._cfg.eval_steps:
                self._evaluate(self._train_set, mode="train")
                self._evaluate(self._test_set, mode="test")
                self._print_metrics()
            if (step + 1) in self._cfg.save_steps:
                self._gaussians.save_std_paramters(
                    os.path.join(self._model_dir, f"{step + 1}.ply")
                )

        self._after_train_loop()

        print(f"Usage time: {usage_time:.2f}s")
        with open(os.path.join(self._save_dir, "usage_time.txt"), "w") as f:
            f.write(f"usage time: {usage_time} sec\n")
