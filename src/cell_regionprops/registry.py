"""Built-in measurement functions for labeled cell masks.

Each property function receives a single binary object mask, the integer label
that mask came from, and the pixel size. Functions return dictionaries so one
logical property can contribute one or more dataframe columns.
"""

from __future__ import annotations

from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

from cell_regionprops.morphometrics import property_morphometrics

PropertyResult: TypeAlias = dict[str, object]
PropertyFunction: TypeAlias = Callable[[NDArray[np.bool_], int, float], PropertyResult]

DEFAULT_PROPERTIES: tuple[str, ...] = ("label", "area", "centroid", "moments_axis")


def property_label(
    mask: NDArray[np.bool_],
    label: int,
    pixel_size: float,
) -> PropertyResult:
    """Return the source label for a mask.

    Args:
        mask: Binary mask for one labeled object.
        label: Integer label represented by `mask`.
        pixel_size: Pixel size in physical units.

    Returns:
        A dictionary containing the `label` column.
    """
    del mask, pixel_size
    return {"label": int(label)}


def property_area(
    mask: NDArray[np.bool_],
    label: int,
    pixel_size: float,
) -> PropertyResult:
    """Measure mask area in pixels and physical units.

    Args:
        mask: Binary mask for one labeled object.
        label: Integer label represented by `mask`.
        pixel_size: Pixel size in physical units.

    Returns:
        A dictionary containing pixel area and physical area.
    """
    del label
    area_px = int(mask.sum())
    return {"area_px": area_px, "area": area_px * pixel_size**2}


def property_centroid(
    mask: NDArray[np.bool_],
    label: int,
    pixel_size: float,
) -> PropertyResult:
    """Measure the centroid of foreground pixels.

    Args:
        mask: Binary mask for one labeled object.
        label: Integer label represented by `mask`.
        pixel_size: Pixel size in physical units.

    Returns:
        A dictionary containing `centroid_y` and `centroid_x`. Empty masks
        return NaN coordinates.
    """
    del label, pixel_size
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return {"centroid_y": np.nan, "centroid_x": np.nan}
    return {"centroid_y": float(rows.mean()), "centroid_x": float(cols.mean())}


def property_moments_axis(
    mask: NDArray[np.bool_],
    label: int,
    pixel_size: float,
) -> PropertyResult:
    """Estimate major and minor axis lengths from mask moments.

    The calculation mirrors the common regionprops-style covariance estimate:
    foreground pixel coordinates are converted into a covariance matrix, and
    the sorted eigenvalues are scaled into major and minor axis lengths.

    Args:
        mask: Binary mask for one labeled object.
        label: Integer label represented by `mask`.
        pixel_size: Pixel size in physical units.

    Returns:
        A dictionary containing moment-derived pixel and physical major/minor
        axis estimates.
    """
    del label
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return {
            "length_px_moments": np.nan,
            "width_px_moments": np.nan,
            "length_moments": np.nan,
            "width_moments": np.nan,
        }

    coordinates = np.column_stack((rows, cols)).astype(float)
    if coordinates.shape[0] == 1:
        eigenvalues = np.array([0.0, 0.0])
    else:
        covariance = np.cov(coordinates, rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]

    length_px = float(4.0 * np.sqrt(eigenvalues[0]))
    width_px = float(4.0 * np.sqrt(eigenvalues[1]))
    return {
        "length_px_moments": length_px,
        "width_px_moments": width_px,
        "length_moments": length_px * pixel_size,
        "width_moments": width_px * pixel_size,
    }


PROPERTY_REGISTRY: dict[str, PropertyFunction] = {
    "label": property_label,
    "area": property_area,
    "centroid": property_centroid,
    "moments_axis": property_moments_axis,
    "morphometrics": property_morphometrics,
}
