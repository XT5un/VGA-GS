import os
from typing import Dict, Literal, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray
from plyfile import PlyData, PlyElement

__all__ = [
    "read_rgb",
    "read_gray",
    "read_pfm",
    "write_rgb",
    "write_rgba",
    "write_gray",
    "write_correspondence",
    "read_points",
    "write_points",
    "read_ply_file",
    "write_ply_file",
]


def read_rgb(path: str, normalization=True) -> ndarray:
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if normalization:
        rgb = rgb.astype(np.float32) / 255.0
    return rgb


def read_gray(path: str, scale: Optional[float] = None) -> ndarray:
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if scale is not None:
        depth /= scale
    return depth


def read_pfm(path: str) -> Tuple[ndarray, float]:
    with open(path, "rb") as fp:
        header = fp.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a valid PFM file.")

        # Read dimensions
        dimensions = fp.readline().decode("utf-8").rstrip()
        width, height = map(int, dimensions.split())

        # Read scale factor
        scale_line = fp.readline().decode("utf-8").rstrip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"

        # Read pixel data
        data = np.fromfile(fp, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(
            data
        )  # Flip vertically because PFM stores pixels from bottom to top
        data = data.copy()  # Make sure the data is contiguous

        return data, scale


def write_rgb(
    path: str,
    image: ndarray,
    mask: ndarray | None = None,
    text: str | None = None,
):
    image = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if mask is not None:
        image[~mask] = (0, 0, 0)
    if text is not None:
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(path, image)


def write_rgba(
    path: str,
    image: ndarray,
    alpha: ndarray | None = None,
):
    ext = os.path.splitext(path)[1]
    if ext.lower() != ".png":
        path = path.replace(ext, ".png")

    if alpha is None:
        alpha = np.ones(image.shape[:2], dtype=image.dtype)
    image = np.concatenate((image, alpha[..., None]), axis=-1)
    image = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, image)


def write_correspondence(
    path: str,
    image0: ndarray,
    image1: ndarray,
    pts0: ndarray,
    pts1: ndarray,
    max_num: int | None = None,
    inlier: ndarray | None = None,
):
    if inlier is None:
        inlier = np.ones(len(pts0), dtype=np.bool_)

    assert len(pts0) == len(pts1)
    W = image0.shape[1]
    merged = np.hstack((image0, image1))
    merged = cv2.cvtColor((merged * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    pts0, pts1 = pts0.astype(np.int32), pts1.astype(np.int32)
    indices = np.arange(len(pts0))
    if max_num is not None and len(pts0) > max_num:
        indices = np.random.choice(indices, max_num, replace=False)
    for i in indices:
        c = (0, 255, 0) if inlier[i] else (0, 0, 255)
        pt0 = pts0[i]
        pt1 = pts1[i]
        pt1[0] += W
        cv2.line(merged, tuple(pt0), tuple(pt1), c, 1)

    cv2.imwrite(path, merged)


def write_gray(path: str, image: ndarray, bits: Literal[8, 16] = 8):
    if image.max() > 1.0:
        image /= image.max()
    image *= 2**bits - 1

    if bits == 8:
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint16)

    cv2.imwrite(path, image)


def read_points(path: str) -> Tuple[ndarray, ndarray]:
    with open(path, "rb") as f:
        ply = PlyData.read(f)
    points = np.stack(
        [ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]], axis=-1
    )
    colors = np.stack(
        [ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]], axis=-1
    )

    points = points.astype(np.float32)
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0
    else:
        colors = colors.astype(np.float32)

    return points, colors


def write_points(path: str, points: ndarray, colors: ndarray | None = None) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        if colors.dtype == np.float32:
            dtype += [("red", "f4"), ("green", "f4"), ("blue", "f4")]
        elif colors.dtype == np.uint8:
            dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        else:
            colors = colors.astype(np.float32)
            dtype += [("red", "f4"), ("green", "f4"), ("blue", "f4")]
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    if colors is not None:
        data["red"] = colors[:, 0]
        data["green"] = colors[:, 1]
        data["blue"] = colors[:, 2]

    ply = PlyData([PlyElement.describe(data, "vertex")])
    ply.write(path)


def read_ply_file(path: str) -> Dict[str, ndarray]:
    with open(path, "rb") as f:
        ply = PlyData.read(f)
    properties = {}
    for key in ply["vertex"].data.dtype.names:
        properties[key] = np.asarray(ply["vertex"][key])
    return properties


def write_ply_file(path: str, properties: Dict[str, ndarray]) -> None:
    dtype = []
    for key, value in properties.items():
        assert type(value) == np.ndarray, f"Value of {key} is not a numpy array"
        assert value.ndim == 1, f"Value of {key} is not a 1D array"
        dtype.append((key, value.dtype))

    data = np.empty(len(properties[list(properties.keys())[0]]), dtype=dtype)
    for key, value in properties.items():
        data[key] = value

    ply = PlyData([PlyElement.describe(data, "vertex")])
    ply.write(path)
