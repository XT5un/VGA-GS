from typing import Tuple

import cv2
import numpy as np
from matplotlib import colormaps
from numpy import ndarray

__all__ = ["colorize", "resize"]


def resize(
    data: ndarray,
    factor: int | None = None,
    dszie: Tuple[int, int] | None = None,
) -> ndarray:
    if dszie is not None:
        return cv2.resize(data, dszie)

    H, W = data.shape[:2]
    H, W = H // factor, W // factor
    return cv2.resize(data, (W, H))


def colorize(
    image: ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: ndarray | None = None,
    color_map: str = "plasma",
    reverse: bool = False,
    margin_thresh: float = 1e-4,
) -> ndarray:
    if mask is not None and mask.sum() == 0:
        H, W = image.shape[:2]
        return np.zeros((H, W, 3), dtype=np.float32)

    if vmin is None:
        vmin = (
            np.percentile(image, 2) if mask is None else np.percentile(image[mask], 2)
        )
    if vmax is None:
        vmax = (
            np.percentile(image, 98) if mask is None else np.percentile(image[mask], 98)
        )

    if (vmax - vmin) < margin_thresh:
        image = np.zeros_like(image)
    else:
        image = (image - vmin) / (vmax - vmin)
    if reverse:
        image = 1.0 - np.clip(image, 0.0, 1.0)
    else:
        image = np.clip(image, 0.0, 1.0)

    cmap = colormaps[color_map]
    image = cmap(image)[:, :, :3]

    if mask is not None:
        image[~mask] = (0.0, 0.0, 0.0)

    return image
