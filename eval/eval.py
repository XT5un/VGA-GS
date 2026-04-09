import argparse
import json
import os
import sys
from typing import List, Literal, Tuple

import torch
from rich import print

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scene.data import GroundTruthDataset
from scene.gaussians import Gaussians3D
from scene.metrics import MultiMetrics
from scene.renderer import RenderResult, renderer_2DGS, renderer_RaDe
from train.base_trainer.configs import load_gin_configs
from train.vga_trainer.configs import VGAGSConfigs
from utils import colorize, write_rgb

_DEVICE = torch.device("cuda")


def _get_configs(workspace: str) -> VGAGSConfigs:
    gin_file = os.path.join(workspace, "configs.gin")
    load_gin_configs(gin_file)
    return VGAGSConfigs()


def _load_model(cfg: VGAGSConfigs, workspace: str, model_iter: int = -1) -> Gaussians3D:
    model_dir = os.path.join(workspace, "models")
    if model_iter == -1:
        iteration = max([int(os.path.splitext(f)[0]) for f in os.listdir(model_dir)])
    else:
        iteration = model_iter
    model_path = os.path.join(model_dir, f"{iteration}.ply")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    gaussians = Gaussians3D(cfg.sh_degree, _DEVICE)
    gaussians.load_std_parameters(model_path)

    return gaussians


def _load_dataset(cfg: VGAGSConfigs) -> Tuple[GroundTruthDataset, GroundTruthDataset]:
    train_set = GroundTruthDataset(
        cfg.dataset_name,
        cfg.data_root_dir,
        cfg.scene_name,
        cfg.resize_factor,
        cfg.n_sparse_views,
        mode="train",
        device=_DEVICE,
        data_device=_DEVICE,
    )
    test_set = GroundTruthDataset(
        cfg.dataset_name,
        cfg.data_root_dir,
        cfg.scene_name,
        cfg.resize_factor,
        cfg.n_sparse_views,
        mode="test",
        device=_DEVICE,
        data_device=_DEVICE,
    )
    return train_set, test_set


@torch.no_grad()
def _render(
    cfg: VGAGSConfigs, model: Gaussians3D, dataset: GroundTruthDataset
) -> List[RenderResult]:
    if cfg.rasterizer_backend == "RaDe":
        renderer = renderer_RaDe
    elif cfg.rasterizer_backend == "2dgs":
        renderer = renderer_2DGS
    else:
        raise ValueError(f"Invalid rasterizer type {cfg.rasterizer_backend}")

    bg_color = torch.tensor(cfg.bg_color, device=_DEVICE)

    rendered_rlts = []
    for cam_id in range(len(dataset)):
        camera = dataset[cam_id]
        render_rlt = renderer(
            model.get_render_parameters(cfg.sh_degree),
            camera.model,
            camera.w2c,
            bg_color,
            False,
            False,
        )
        rendered_rlts.append(render_rlt)

    return rendered_rlts


def _eval_images(
    dataset: GroundTruthDataset, rendered_rlts: List[RenderResult]
) -> dict:
    metrics = MultiMetrics(_DEVICE, lpips_net_type="vgg")
    metrics_alex = MultiMetrics(_DEVICE, lpips_net_type="alex")

    metrics_per_view = []
    for camera_id in range(len(dataset)):
        camera = dataset[camera_id]
        rendered_rlt = rendered_rlts[camera_id]

        assert camera.image is not None
        psnr = metrics.compute_psnr(rendered_rlt.image, camera.image, camera.image_mask)
        ssim = metrics.compute_ssim(rendered_rlt.image, camera.image, camera.image_mask)
        ssim_sk = metrics.compute_ssim_sk(
            rendered_rlt.image, camera.image, camera.image_mask
        )
        lpips_vgg = metrics.compute_lpips(
            rendered_rlt.image, camera.image, camera.image_mask
        )
        lpips_alex = metrics_alex.compute_lpips(
            rendered_rlt.image, camera.image, camera.image_mask
        )

        if camera.depth is not None:
            rmse = metrics.compute_depth_rmse(
                rendered_rlt.depth, camera.depth, camera.depth_mask
            )
        else:
            rmse = 0.0

        metrics_per_view.append(
            {
                "name": camera.name,
                "psnr": psnr,
                "ssim": ssim,
                "ssim_sk": ssim_sk,
                "lpips_vgg": lpips_vgg,
                "lpips_alex": lpips_alex,
                "rmse": rmse,
            }
        )

    mean_metrics = {
        "psnr": sum(r["psnr"] for r in metrics_per_view) / len(metrics_per_view),
        "ssim": sum(r["ssim"] for r in metrics_per_view) / len(metrics_per_view),
        "ssim_sk": sum(r["ssim_sk"] for r in metrics_per_view) / len(metrics_per_view),
        "lpips_vgg": sum(r["lpips_vgg"] for r in metrics_per_view)
        / len(metrics_per_view),
        "lpips_alex": sum(r["lpips_alex"] for r in metrics_per_view)
        / len(metrics_per_view),
        "rmse": sum(r["rmse"] for r in metrics_per_view) / len(metrics_per_view),
    }
    meta = {
        "frames": metrics_per_view,
        "mean_metrics": mean_metrics,
    }
    return meta


def _save_visualization(
    workspace: str,
    dataset: GroundTruthDataset,
    rendered_rlts: List[RenderResult],
    mode: Literal["train", "test"],
) -> None:
    save_dir = os.path.join(workspace, f"vis_{mode}")
    os.makedirs(save_dir, exist_ok=True)
    for camera_id in range(len(dataset)):
        camera = dataset[camera_id]
        rendered_rlt = rendered_rlts[camera_id]
        name = camera.name

        assert camera.image is not None
        gt_image = camera.image.cpu().numpy()
        gt_depth = camera.depth.cpu().numpy() if camera.depth is not None else None
        gt_depth_mask = (
            camera.depth_mask.cpu().numpy() if camera.depth_mask is not None else None
        )
        pred_image = rendered_rlt.image.cpu().numpy()
        pred_depth = rendered_rlt.depth.cpu().numpy()

        # colorize depth
        if gt_depth is not None:
            gt_depth = colorize(gt_depth, mask=gt_depth_mask)
        pred_depth = colorize(pred_depth, mask=pred_depth > 0)

        # save images
        write_rgb(os.path.join(save_dir, f"{name}_image_gt.png"), gt_image)
        write_rgb(os.path.join(save_dir, f"{name}_image_render.png"), pred_image)
        if gt_depth is not None:
            write_rgb(os.path.join(save_dir, f"{name}_depth_gt.png"), gt_depth)
        write_rgb(os.path.join(save_dir, f"{name}_depth_render.png"), pred_depth)


def _save_metrics(workspace: str, meta: dict, mode: Literal["train", "test"]) -> None:
    metrics_path = os.path.join(workspace, f"metrics_{mode}.json")
    with open(metrics_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(meta)


def main(workspace: str) -> None:
    if not os.path.exists(workspace):
        raise FileNotFoundError(f"Workspace directory {workspace} not found")

    cfg = _get_configs(workspace)

    gaussians = _load_model(cfg, workspace)
    train_set, test_set = _load_dataset(cfg)

    train_rlts = _render(cfg, gaussians, train_set)
    train_metrics = _eval_images(train_set, train_rlts)
    _save_metrics(workspace, train_metrics, "train")
    _save_visualization(workspace, train_set, train_rlts, "train")

    test_rlts = _render(cfg, gaussians, test_set)
    test_metrics = _eval_images(test_set, test_rlts)
    _save_metrics(workspace, test_metrics, "test")
    _save_visualization(workspace, test_set, test_rlts, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=str, required=True, help="Workspace directory"
    )
    args = parser.parse_args()
    main(args.workspace)
