import json
import os
from typing import Any, Dict, List, Literal, TypeAlias

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from utils import read_gray, read_rgb, resize

from .camera import BaseCamera, DataCamera, PinholeModel


def _opengl2opencv(rotmat: np.ndarray) -> np.ndarray:
    """convert opengl-style pose to opencv-style pose"""
    trans = np.diag([1, -1, -1]).astype(rotmat.dtype)
    return rotmat @ trans


def _load_data(
    data_dir: str,
    device: torch.device,
    data_device: torch.device,
    factor: int = 1,
    **kwargs,
) -> List[DataCamera]:
    cameras: List[DataCamera] = []

    image_dir = os.path.join(data_dir, "images")
    for fname in os.listdir(image_dir):
        image_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        image = read_rgb(image_path)
        if factor > 1:
            image = resize(image, factor)
        image = torch.FloatTensor(image)

        H, W = image.shape[:2]
        focal_length = max(H, W) * 0.75
        principal_x = W * 0.5
        principal_y = H * 0.5
        camera_model = PinholeModel(
            W, H, focal_length, focal_length, principal_x, principal_y, device
        )

        pose = torch.eye(4, dtype=torch.float32)
        camera = DataCamera(pose, camera_model, device, data_device, image_name)
        camera.image = image

        cameras.append(camera)

    return cameras


def _load_ScanNet_DDP_data(
    data_dir: str,
    mode: str,
    device: torch.device,
    data_device: torch.device,
    factor: int = 1,
    **kwargs,
) -> List[DataCamera]:

    cameras: List[DataCamera] = []

    with open(os.path.join(data_dir, f"transforms_{mode}.json"), "r") as f:
        meta = json.load(f)

    depth_scale = meta["depth_scaling_factor"]

    for id, frame in enumerate(meta["frames"]):
        fx, fy = frame["fx"], frame["fy"]
        cx, cy = frame["cx"], frame["cy"]
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, :3] = _opengl2opencv(c2w[:3, :3])
        c2w = torch.FloatTensor(c2w)
        if factor > 1:
            fx, fy = fx / factor, fy / factor
            cx, cy = cx / factor, cy / factor

        if mode in ["train", "test"]:
            image_path = os.path.join(data_dir, frame["file_path"])
            depth_path = os.path.join(
                data_dir, frame["depth_file_path"].replace("depth", "target_depth")
            )
            image = read_rgb(image_path)
            depth = read_gray(depth_path, depth_scale)
            if factor > 1:
                image = resize(image, factor)
                depth = resize(depth, factor)

            image = torch.FloatTensor(image)
            depth = torch.FloatTensor(depth)
            depth_mask = torch.BoolTensor(depth > 0.1)
            name = os.path.splitext(os.path.basename(image_path))[0]
            H, W = image.shape[:2]
        else:
            raise NotImplementedError

        camera_model = PinholeModel(W, H, fx, fy, cx, cy, device)
        camera = DataCamera(c2w, camera_model, device, data_device, name)
        camera.image = image
        camera.depth = depth
        camera.depth_mask = depth_mask
        cameras.append(camera)

    return cameras


def _load_Replica_data(
    data_dir: str,
    mode: str,
    device: torch.device,
    data_device: torch.device,
    factor: int = 1,
    **kwargs,
) -> List[DataCamera]:
    if mode == "train":
        n_sparse_views = kwargs.get("n_sparse_views", -1)
    else:
        n_sparse_views = -1

    cameras: List[DataCamera] = []

    with open(os.path.join(data_dir, f"transforms_{mode}.json"), "r") as f:
        meta = json.load(f)

    depth_scale = meta["depth_scaling_factor"]

    if n_sparse_views > 0 and mode == "train":
        used_ids = np.linspace(0, len(meta["frames"]) - 1, n_sparse_views, dtype=int)
        frames = [meta["frames"][i] for i in used_ids]
    else:
        frames = meta["frames"]

    for frame in frames:
        fx, fy = frame["fx"], frame["fy"]
        cx, cy = frame["cx"], frame["cy"]
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, :3] = _opengl2opencv(c2w[:3, :3])
        c2w = torch.FloatTensor(c2w)
        if factor > 1:
            fx, fy = fx / factor, fy / factor
            cx, cy = cx / factor, cy / factor

        if mode in ["train", "test"]:
            image_path = os.path.join(data_dir, frame["file_path"])
            depth_path = os.path.join(
                data_dir,
                frame["file_path"].replace("rgb", "target_depth").replace("jpg", "png"),
            )
            image = read_rgb(image_path)
            depth = read_gray(depth_path, depth_scale)
            if factor > 1:
                image = resize(image, factor)
                depth = resize(depth, factor)

            image = torch.FloatTensor(image)
            depth = torch.FloatTensor(depth)
            depth_mask = torch.BoolTensor(depth > 0.1)
            name = os.path.splitext(os.path.basename(image_path))[0]
            H, W = image.shape[:2]
        else:
            raise NotImplementedError

        camera_model = PinholeModel(W, H, fx, fy, cx, cy, device)
        camera = DataCamera(c2w, camera_model, device, data_device, name)
        camera.image = image
        camera.depth = depth
        camera.depth_mask = depth_mask
        cameras.append(camera)

    return cameras


def _load_nrgbd_data(
    data_dir: str,
    mode: str,
    device: torch.device,
    data_device: torch.device,
    factor: int = 1,
    **kwargs,
) -> List[DataCamera]:
    cameras: List[DataCamera] = []

    depth_scale = 1000.0
    focal = float(np.loadtxt(os.path.join(data_dir, "focal.txt")))
    num_views = len(os.listdir(os.path.join(data_dir, "images")))
    # 从所有视图中选择20个训练视角和20个测试视角
    used_ids = np.linspace(0, num_views - 1, 20 * 2, dtype=np.int32)
    if mode == "train":
        used_ids = used_ids[::2]
    elif mode == "test":
        used_ids = used_ids[1::2]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    all_poses = np.loadtxt(os.path.join(data_dir, "poses.txt"))
    all_poses = all_poses.reshape(num_views, 4, 4)

    for frame_id in used_ids:
        c2w = all_poses[frame_id]
        c2w[:3, :3] = _opengl2opencv(c2w[:3, :3])
        c2w = torch.FloatTensor(c2w)

        image_path = os.path.join(data_dir, "images", f"img{frame_id}.png")
        depth_path = os.path.join(data_dir, "depth_gt", f"depth{frame_id}.png")
        if not os.path.exists(depth_path):
            depth_path = os.path.join(
                data_dir, "depth_filtered", f"depth{frame_id}.png"
            )

        image = read_rgb(image_path)
        depth = read_gray(depth_path, depth_scale)
        H, W = image.shape[:2]
        fx, fy = focal, focal
        cx, cy = W / 2.0, H / 2.0
        if factor > 1:
            image = resize(image, factor)
            depth = resize(depth, factor)
            fx, fy = fx / factor, fy / factor
            cx, cy = cx / factor, cy / factor
            H, W = image.shape[:2]

        image = torch.FloatTensor(image)
        depth = torch.FloatTensor(depth)
        depth_mask = torch.BoolTensor(depth > 0.1)
        name = os.path.splitext(os.path.basename(image_path))[0]

        camera_model = PinholeModel(W, H, fx, fy, cx, cy, device)
        camera = DataCamera(c2w, camera_model, device, data_device, name)
        camera.image = image
        camera.depth = depth
        camera.depth_mask = depth_mask
        cameras.append(camera)

    return cameras


class Dataset:
    """Dataset base class. It is used to load and preprocess the dataset without GT labels."""

    def __init__(
        self,
        data_root_dir: str,
        scene_name: str,
        resize_factor: int = 1,
        device: str | torch.device = "cuda",
        data_device: str | torch.device = "cuda",
    ) -> None:
        self._scene_name = scene_name
        self._data_dir = os.path.join(data_root_dir, scene_name)
        if not os.path.exists(self._data_dir):
            raise ValueError(f"Data directory {self._data_dir} does not exist.")

        self._resize_factor = resize_factor
        if not isinstance(self._resize_factor, int) or self._resize_factor < 1:
            raise ValueError("Invalid resize factor. It should be an integer >= 1.")

        self._device = torch.device(device) if isinstance(device, str) else device
        self._data_device = (
            torch.device(data_device) if isinstance(data_device, str) else data_device
        )

        self._cameras: List[DataCamera] = []

        self._load_data()
        self._name2id = {
            camera.name: cam_id for cam_id, camera in enumerate(self._cameras)
        }

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def resize_factor(self) -> int:
        return self._resize_factor

    @property
    def name2id(self) -> Dict[str, int]:
        return self._name2id

    def __len__(self) -> int:
        return len(self._cameras)

    def __getitem__(self, index: int) -> DataCamera:
        return self._cameras[index]

    def _load_data(self) -> None:
        self._cameras = _load_data(
            self._data_dir, self._device, self._data_device, self._resize_factor
        )

    def save_json(self, path: str) -> None:
        meta = []
        for cam_id, camera in enumerate(self._cameras):
            camera_dict = {
                "id": cam_id,
                "img_name": camera.name,
                "width": camera.model.width,
                "height": camera.model.height,
                "position": camera.w2c[:3, 3].cpu().numpy().tolist(),
                "rotation": camera.w2c[:3, :3].cpu().numpy().tolist(),
                "fx": camera.model.fx,
                "fy": camera.model.fy,
            }
            meta.append(camera_dict)

        with open(path, "w") as f:
            json.dump(meta, f, indent=4)


SUPPORTED_DATASETS: TypeAlias = Literal[
    "ScanNet",
    "Replica",
    "NRGBD",
]

_DATASET_LOADER: Dict[SUPPORTED_DATASETS, Any] = {
    "ScanNet": _load_ScanNet_DDP_data,
    "NRGBD": _load_nrgbd_data,
    "Replica": _load_Replica_data,
}


class GroundTruthDataset(Dataset):
    """Dataset class for training data with GT labels."""

    def __init__(
        self,
        dataset_name: SUPPORTED_DATASETS,
        data_root_dir: str,
        scene_name: str,
        resize_factor: int = 1,
        n_sparse_views: int = -1,
        mode: str = "train",
        device: str | torch.device = "cuda",
        data_device: str | torch.device = "cuda",
    ) -> None:
        self._dataset_name = dataset_name
        self._mode = mode
        self._n_sparse_views = n_sparse_views

        if dataset_name != "Replica" and n_sparse_views != -1:
            raise ValueError(
                f"`n_sparse_views` is only supported for Replica dataset, got {dataset_name}."
            )

        super().__init__(data_root_dir, scene_name, resize_factor, device, data_device)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _load_data(self) -> None:
        if self._dataset_name not in _DATASET_LOADER:
            raise ValueError(f"Unsupported Dataset: {self._dataset_name}")

        data_loader_func = _DATASET_LOADER[self._dataset_name]

        self._cameras = data_loader_func(
            self._data_dir,
            self._mode,
            self._device,
            self._data_device,
            self._resize_factor,
            n_sparse_views=self._n_sparse_views,
        )


def get_scene_radius(dataset: Dataset) -> float:
    cam_centers = [cam.c2w[:3, 3] for cam in dataset]
    cam_centers = torch.stack(cam_centers, dim=0)
    mean_center = cam_centers.mean(dim=0, keepdim=True)
    dists = torch.norm(cam_centers - mean_center, dim=1)
    diagonal = torch.max(dists).item()
    radius = diagonal * 1.1
    return radius


def get_scene_scale(dataset: Dataset) -> float:
    cam_positions = [cam.c2w[:3, 3] for cam in dataset]
    cam_positions = torch.stack(cam_positions, dim=0)
    # get matrix of pairwise distances
    dists = torch.cdist(cam_positions, cam_positions)
    # mask the diagonal and get the min distance
    dists.fill_diagonal_(float("inf"))
    min_dist = torch.min(dists, dim=1).values
    scale = torch.mean(min_dist).item()
    return scale


def _generate_spiral_trajectory(dataset: Dataset, n_frames: int) -> List[BaseCamera]:
    """Ref: https://github.com/jiaw-z/CoR-GS/blob/86ffb3e61aaf4a09f1e75d968af244f6bdee0540/scene/dataset_readers.py#L643"""

    def normalize(x):
        return x / np.linalg.norm(x)

    def viewmatrix(lookdir, up, position, subtract_position=False):
        """Construct lookat view matrix."""
        vec2 = normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def poses_avg(poses):
        """New pose using average position, z-axis, and up vector of input poses."""
        position = poses[:, :3, 3].mean(0)
        z_axis = poses[:, :3, 2].mean(0)
        up = poses[:, :3, 1].mean(0)
        cam2world = viewmatrix(z_axis, up, position)
        return cam2world

    def pad_poses(p):
        """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
        bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
        return np.concatenate([p[..., :3, :4], bottom], axis=-2)

    def unpad_poses(p):
        """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
        return p[..., :3, :4]

    def recenter_poses(poses):
        """Recenter poses around the origin."""
        cam2world = poses_avg(poses)
        poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
        return unpad_poses(poses)

    def backcenter_poses(poses, pose_ref):
        """Recenter poses around the origin."""
        cam2world = poses_avg(pose_ref)
        poses = pad_poses(cam2world) @ pad_poses(poses)
        return unpad_poses(poses)

    def focus_pt_fn(poses):
        """Calculate nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt

    def generate_spiral_path(poses, n_frames=120, n_rots=2, zrate=0.5, perc=60):
        """Calculates a forward facing spiral path for rendering for DTU."""

        # Get radii for spiral path using 60th percentile of camera positions.
        positions = poses[:, :3, 3]
        radii = np.percentile(np.abs(positions), perc, 0)
        radii = np.concatenate([radii, [1.0]])

        # Generate poses for spiral path.
        render_poses = []
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        z_axis = focus_pt_fn(poses)
        for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
            t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
            position = cam2world @ t
            # print(f"position is {position}")
            render_poses.append(viewmatrix(z_axis, up, position, True))
        render_poses = np.stack(render_poses, axis=0)
        return render_poses

    poses_arr = np.stack([cam.c2w.cpu().numpy() for cam in dataset])
    poses_o = poses_arr[:, :3, :4]
    fix_rotation = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    inv_rotation = np.linalg.inv(fix_rotation)
    poses = poses_o @ fix_rotation

    render_poses = recenter_poses(poses)

    s = np.max(np.abs(render_poses[:, :3, -1]))
    render_poses[:, :3, -1] /= s

    render_poses = generate_spiral_path(render_poses, n_frames=n_frames)

    render_poses[:, :3, -1] *= s
    render_poses = backcenter_poses(render_poses, poses)

    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate(
        [render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1
    )
    render_poses = pad_poses(render_poses)

    camera_model = dataset[0].model
    device = dataset[0].device
    render_cameras = []
    for pose in render_poses:
        c2w = torch.FloatTensor(pose).to(device)
        camera = BaseCamera(c2w, camera_model, device)
        render_cameras.append(camera)

    return render_cameras


def _generate_sequence_trajectory(
    dataset: Dataset, n_frames_between_keyframes: int
) -> List[BaseCamera]:
    poses = np.stack([cam.c2w.cpu().numpy() for cam in dataset])
    render_poses = []
    for curr_i in range(1, len(poses)):
        prev_i = curr_i - 1
        prev_pose = poses[prev_i]
        curr_pose = poses[curr_i]

        w = np.linspace(0, 1, n_frames_between_keyframes + 1, endpoint=True)
        if curr_i != len(poses) - 1:
            w = w[:-1]
        lerp_pos = prev_pose[:3, 3] * (1 - w[:, None]) + curr_pose[:3, 3] * w[:, None]

        rot = Rotation.from_matrix(np.stack([prev_pose[:3, :3], curr_pose[:3, :3]]))
        slerp = Slerp([0.0, 1.0], rot)
        slerp_rot = slerp(w).as_matrix()

        lerp_poses = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(len(w), axis=0)
        lerp_poses[:, :3, :3] = slerp_rot
        lerp_poses[:, :3, 3] = lerp_pos.reshape(-1, 3)

        render_poses.append(lerp_poses)

    render_poses = np.concatenate(render_poses, axis=0)
    camera_model = dataset[0].model
    device = dataset[0].device
    render_cameras = []
    for pose in render_poses:
        c2w = torch.FloatTensor(pose).to(device)
        camera = BaseCamera(c2w, camera_model, device)
        render_cameras.append(camera)

    return render_cameras


def get_render_cameras(
    dataset: Dataset, dataset_name: Literal["ScanNet", "Replica", "NRGBD"]
) -> List[BaseCamera]:
    if dataset_name in ["ScanNet", "Replica", "NRGBD"]:
        return _generate_sequence_trajectory(dataset, n_frames_between_keyframes=50)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
