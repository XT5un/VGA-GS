import argparse
import json
import os
import sys
import time
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import trimesh
from rich import print
from scipy.spatial import cKDTree as KDTree

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from scene.data import GroundTruthDataset
from scene.gaussians import Gaussians3D
from scene.renderer import RenderResult, renderer_2DGS, renderer_RaDe
from train.base_trainer.configs import load_gin_configs
from train.vga_trainer.configs import VGAGSConfigs
from utils import TSDFMeshExtrator

_DEVICE = torch.device("cuda")


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace directory for GS training",
    )
    parser.add_argument(
        "--mesh_res",
        type=int,
        default=1024,
        help="Parameter for mesh extraction",
    )
    parser.add_argument(
        "--clean_depth",
        action="store_true",
        help="Clean depth map",
    )
    return parser.parse_args()


def _get_configs(workspace: str) -> VGAGSConfigs:
    gin_file = os.path.join(workspace, "configs.gin")
    load_gin_configs(gin_file)
    return VGAGSConfigs()


def _load_model(cfg: VGAGSConfigs, workspace: str, model_iter: int = -1) -> Gaussians3D:
    model_dir = os.path.join(workspace, "models")
    if model_iter == -1:
        iteration = max([int(os.path.splitext(f)[0]) for f in os.listdir(model_dir)])
    model_path = os.path.join(model_dir, f"{iteration}.ply")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    gaussians = Gaussians3D(cfg.sh_degree, _DEVICE)
    gaussians.load_std_parameters(model_path)

    return gaussians


def _load_dataset(cfg: VGAGSConfigs) -> GroundTruthDataset:
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
    return train_set


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


def extract_mesh(
    mesh_rec_path: str,
    mesh_gt_path: str,
    dataset: GroundTruthDataset,
    rendered_rlts: List[RenderResult],
    mesh_res: int = 1024,
    clean_depth: bool = True,
) -> None:

    depth_trunc = -float("inf")
    for camera_id in range(len(dataset)):
        camera = dataset[camera_id]
        assert camera.depth is not None
        far = torch.max(camera.depth[camera.depth_mask]).item()
        depth_trunc = max(depth_trunc, far)
    depth_trunc *= 1.2

    print(f"Depth truncation: {depth_trunc:.4f}")

    if os.path.exists(mesh_rec_path):
        print(f"Mesh {mesh_rec_path} already exists")
    else:
        tsdf = TSDFMeshExtrator(
            depth_trunc=depth_trunc, mesh_res=mesh_res, clean_depth=clean_depth
        )
        for camera_id in range(len(dataset)):
            camera = dataset[camera_id]
            rendered_rlt = rendered_rlts[camera_id]
            tsdf.add_frame(rendered_rlt.image, rendered_rlt.depth, camera)
        tsdf.extract_and_save(mesh_rec_path)
        print(f"Mesh saved to {mesh_rec_path}")

    if os.path.exists(mesh_gt_path):
        print(f"Ground truth mesh {mesh_gt_path} already exists")
    else:
        tsdf = TSDFMeshExtrator(depth_trunc=depth_trunc, mesh_res=mesh_res)
        for camera_id in range(len(dataset)):
            camera = dataset[camera_id]
            assert camera.depth is not None
            assert camera.depth_mask is not None
            assert camera.image is not None
            gt_depth = camera.depth.clone()
            gt_depth[~camera.depth_mask] = 0.0
            tsdf.add_frame(camera.image, gt_depth, camera)
        tsdf.extract_and_save(mesh_gt_path)
        print(f"Ground truth mesh saved to {mesh_gt_path}")


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc, distances


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp, distances


def get_mesh_metrics(gt_mesh_path: str, rec_mesh_path: str) -> Dict[
    Literal["accuracy", "completion", "completion_ratio", "precision_ratio", "fscore"],
    float,
]:
    # ! Reference: https://github.com/autonomousvision/monosdf/blob/12513009cd20a08b35ddc2c55a7e42da2c7baf43/replica_eval/eval_recon.py#L109

    mesh_gt = trimesh.load(gt_mesh_path, process=False)
    mesh_rec = trimesh.load(rec_mesh_path, process=False)

    if mesh_rec.vertices.shape[0] == 0:
        return {
            "accuracy": 0.0,
            "completion": 0.0,
            "completion_ratio": 0.0,
            "precision_ratio": 0.0,
            "fscore": 0.0,
        }

    # to_align, _ = trimesh.bounds.oriented_bounds(mesh_gt)
    # mesh_gt.vertices = (to_align[:3, :3] @ mesh_gt.vertices.T + to_align[:3, 3:4]).T
    # mesh_rec.vertices = (to_align[:3, :3] @ mesh_rec.vertices.T + to_align[:3, 3:4]).T

    min_points = mesh_gt.vertices.min(axis=0) * 1.05
    max_points = mesh_gt.vertices.max(axis=0) * 1.05

    mask_min = (mesh_rec.vertices - min_points[None]) > 0
    mask_max = (mesh_rec.vertices - max_points[None]) < 0

    mask = np.concatenate((mask_min, mask_max), axis=1).all(axis=1)
    face_mask = mask[mesh_rec.faces].all(axis=1)

    mesh_rec.update_vertices(mask)
    mesh_rec.update_faces(face_mask)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    accuracy_rec, dist_d2s = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec, dist_s2d = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)
    precision_ratio_rec = completion_ratio(rec_pc_tri.vertices, gt_pc_tri.vertices)
    fscore = (
        2
        * precision_ratio_rec
        * completion_ratio_rec
        / (completion_ratio_rec + precision_ratio_rec)
    )

    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    precision_ratio_rec *= 100  # convert to %
    fscore *= 100

    meta = {
        "accuracy": accuracy_rec,
        "completion": completion_rec,
        "completion_ratio": completion_ratio_rec,
        "precision_ratio": precision_ratio_rec,
        "fscore": fscore,
    }

    return meta


def main(args) -> None:
    if not os.path.exists(args.workspace):
        raise FileNotFoundError(f"Workspace directory {args.workspace} not found")

    cfg = _get_configs(args.workspace)

    model = _load_model(cfg, args.workspace)
    dataset = _load_dataset(cfg)
    rendered_rlts = _render(cfg, model, dataset)

    metrics_path = os.path.join(args.workspace, "metrics_recon.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        for k, v in metrics.items():
            if v > 0.0:
                print(f"metrics file {metrics_path} already exists")
                sys.exit(0)

    mesh_gt_path = os.path.join(args.workspace, "mesh_gt.ply")
    mesh_rec_path = os.path.join(args.workspace, "mesh_recon.ply")
    extract_mesh(
        mesh_rec_path,
        mesh_gt_path,
        dataset,
        rendered_rlts,
        args.mesh_res,
        args.clean_depth,
    )
    metrics = get_mesh_metrics(mesh_gt_path, mesh_rec_path)

    print(metrics)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    args = _get_args()
    main(args)
