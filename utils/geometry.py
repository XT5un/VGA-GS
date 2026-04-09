from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
from numpy import ndarray

__all__ = [
    "refine_correspondence_by_mono_depth",
    "solve_pose_PnP",
    "triangulate_points",
    "clean_depthmap",
]


def clean_depthmap(
    depthmap: ndarray,
    K: ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    num_downsample: int | None = None,
    min_depth: float = 0.01,
) -> Tuple[ndarray, ndarray]:
    """Denoise for depthmap"""

    pointmap = _depth2points(depthmap, K)
    valid_mask = np.zeros_like(depthmap, dtype=np.bool_)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointmap.reshape(-1, 3))
    pcd, idxs = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    idxs = np.asarray(idxs)
    if num_downsample is not None and num_downsample > 0:
        idxs = np.random.choice(idxs, num_downsample, replace=False)

    if len(idxs) == 0:
        return depthmap, np.ones_like(depthmap, dtype=np.bool_)

    valid_mask = valid_mask.reshape(-1)
    valid_mask[idxs] = True
    valid_mask = valid_mask.reshape(depthmap.shape)
    valid_mask = valid_mask & (depthmap > min_depth)

    cleaned_depthmap = depthmap.copy()
    cleaned_depthmap[~valid_mask] = 0

    return cleaned_depthmap, valid_mask


def triangulate_points(
    pts0: ndarray, pts1: ndarray, K0: ndarray, K1: ndarray, P0: ndarray, P1: ndarray
) -> ndarray:
    prjmat0 = K0 @ P0[:3]
    prjmat1 = K1 @ P1[:3]
    pts4d = cv2.triangulatePoints(
        prjmat0,
        prjmat1,
        pts0.T.astype(np.float64),
        pts1.T.astype(np.float64),
    )
    pts4d = pts4d.astype(np.float32)
    pts3d = (pts4d[:3] / pts4d[3]).T

    return pts3d


def solve_pose_PnP(
    pts3d: ndarray, pts2d, K: ndarray
) -> Tuple[bool, ndarray, ndarray, ndarray]:
    success, rvecs, tvecs, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64),
        pts2d.astype(np.float64),
        K,
        None,
    )
    rotmat, _ = cv2.Rodrigues(rvecs)

    return success, rotmat, tvecs.reshape(3), inliers[:, 0]


def _depth2points(depth: ndarray, K: ndarray) -> ndarray:
    H, W = depth.shape[:2]
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    x = (x.astype(depth.dtype) + 0.5) * depth
    y = (y.astype(depth.dtype) + 0.5) * depth
    z = depth
    pts3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pts3d = np.dot(pts3d, np.linalg.inv(K).T).reshape(H, W, 3)
    return pts3d


def refine_correspondence_by_mono_depth(
    kpts0: ndarray,
    kpts1: ndarray,
    K0: ndarray,
    K1: ndarray,
    depth0: ndarray,
    depth1: ndarray,
    align_depth_threshold: float = 0.05,
    align_ratio_threshold: float = 0.5,
    align_num_threshold: int = 5_000,
    refine_keypoints: bool = True,
):
    pts3d0 = _depth2points(depth0, K0)
    pts3d1 = _depth2points(depth1, K1)
    kpts0_int = kpts0.astype(np.int32)
    kpts1_int = kpts1.astype(np.int32)

    kpts3d0 = pts3d0[kpts0_int[:, 1], kpts0_int[:, 0]]
    kpts3d1 = pts3d1[kpts1_int[:, 1], kpts1_int[:, 0]]

    # 用第一个相机作为世界坐标系，计算第二个相机的pose
    success, R, t, inliers = solve_pose_PnP(kpts3d0, kpts1, K1)
    if not success:
        return False, None, None

    if not refine_keypoints:
        return True, kpts0[inliers], kpts1[inliers]

    kpts3d0 = kpts3d0[inliers]
    kpts3d1 = kpts3d1[inliers]

    # align the 3d points
    kpts3d0_new = np.dot(kpts3d0, R.T) + t
    shift = np.median(kpts3d1 - kpts3d0_new, axis=0)

    # * 这里aligned_pts3d0是第一个相机的投影到第二个相机的3d坐标，所以aligned_pts3d0和pts3d1是对齐的
    aligned_pts3d = np.dot(pts3d0, R.T) + t + shift
    aligned_prjs = np.dot(aligned_pts3d, K1.T)
    aligned_depth = aligned_prjs[..., 2]
    aligned_u = aligned_prjs[..., 0] / np.clip(np.abs(aligned_depth), 1e-6, None)
    aligned_v = aligned_prjs[..., 1] / np.clip(np.abs(aligned_depth), 1e-6, None)
    visiable_mask = (
        (aligned_u >= 0)
        & (aligned_u < depth1.shape[1])
        & (aligned_v >= 0)
        & (aligned_v < depth1.shape[0])
        & (aligned_depth > 0)
    )
    aligned_pts3d = aligned_pts3d[visiable_mask]
    aligned_depth = aligned_depth[visiable_mask]
    aligned_u = aligned_u[visiable_mask]
    aligned_v = aligned_v[visiable_mask]
    u0, v0 = np.meshgrid(
        np.arange(depth0.shape[1]) + 0.5,
        np.arange(depth0.shape[0]) + 0.5,
        indexing="xy",
    )
    u0 = u0[visiable_mask]
    v0 = v0[visiable_mask]

    depth_diff = np.abs(
        aligned_depth - depth1[aligned_v.astype(np.int32), aligned_u.astype(np.int32)]
    )
    refine_mask = depth_diff < align_depth_threshold
    refine_num = refine_mask.sum().item()
    refine_ratio = refine_num / len(depth_diff)
    if refine_num < align_num_threshold or refine_ratio < align_ratio_threshold:
        return False, None, None

    refined_kpts0 = np.stack([u0[refine_mask], v0[refine_mask]], axis=-1)
    refined_kpts1 = np.stack([aligned_u[refine_mask], aligned_v[refine_mask]], axis=-1)

    return True, refined_kpts0.astype(np.float32), refined_kpts1.astype(np.float32)
