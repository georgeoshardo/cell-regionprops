"""napari-MM3 Feret-style geometry for cell masks.

The endpoint search follows napari-MM3's BSD-3-Clause `feretdiameter`
implementation, but this module takes plain mask/moment inputs instead of a
`skimage.measure.RegionProperties` object.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi


def orientation_from_central_moments(
    var_x: Any,
    var_y: Any,
    cov_xy: Any,
) -> Any:
    """Return the same 2D orientation convention used by scikit-image."""
    orientation = 0.5 * np.arctan2(2.0 * cov_xy, var_y - var_x)
    fallback = np.where(cov_xy > 0.0, math.pi / 4.0, -math.pi / 4.0)
    return np.where(var_x == var_y, fallback, orientation)


def mm3_capsule_volume(length: float, width: float) -> float:
    """Compute MM3 capsule volume from length and width."""
    radius = width / 2.0
    return math.pi * radius**2 * (length - width) + (4.0 / 3.0) * math.pi * radius**3


def mm3_capsule_surface_area(length: float, width: float) -> float:
    """Compute MM3 capsule surface area from length and width."""
    return math.pi * length * width


def property_mm3_feret(
    mask: NDArray[np.bool_],
    label: int,
    pixel_size: float,
) -> dict[str, object]:
    """Measure napari-MM3 Feret-style length and width for one object mask."""
    del label
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return _failed_result()

    centroid_y = float(rows.mean())
    centroid_x = float(cols.mean())
    var_x = float(np.mean((cols - centroid_x) ** 2))
    var_y = float(np.mean((rows - centroid_y) ** 2))
    cov_xy = float(np.mean((cols - centroid_x) * (rows - centroid_y)))
    eigenvalue_gap = math.sqrt((var_x - var_y) ** 2 + 4.0 * cov_xy**2)
    major_eigenvalue = max((var_x + var_y + eigenvalue_gap) / 2.0, 0.0)
    minor_eigenvalue = max((var_x + var_y - eigenvalue_gap) / 2.0, 0.0)
    orientation = float(orientation_from_central_moments(var_x, var_y, cov_xy))
    return measure_mm3_feret_from_mask(
        mask,
        pixel_size=pixel_size,
        centroid_y=centroid_y,
        centroid_x=centroid_x,
        orientation=orientation,
        major_axis_length=4.0 * math.sqrt(major_eigenvalue),
        minor_axis_length=4.0 * math.sqrt(minor_eigenvalue),
    )


def measure_mm3_feret_from_mask(
    mask: NDArray[np.bool_],
    *,
    pixel_size: float,
    centroid_y: float,
    centroid_x: float,
    orientation: float,
    major_axis_length: float,
    minor_axis_length: float,
) -> dict[str, object]:
    """Measure MM3 Feret geometry from a mask and precomputed moments."""
    mask_np = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(mask_np)
    if rows.size == 0:
        return _failed_result()

    min_row = int(rows.min())
    max_row = int(rows.max())
    min_col = int(cols.min())
    max_col = int(cols.max())
    cropped = mask_np[min_row:max_row + 1, min_col:max_col + 1]
    y0 = float(centroid_y - min_row + 1.0)
    x0 = float(centroid_x - min_col + 1.0)

    boundary_coords = _boundary_coordinates(cropped)
    if boundary_coords.shape[0] == 0:
        return _failed_result()

    length_px = _mm3_length(
        boundary_coords,
        y0=y0,
        x0=x0,
        orientation=orientation,
        major_axis_length=float(major_axis_length),
    )
    if not np.isfinite(length_px):
        return _failed_result()

    width_px = _mm3_width(
        boundary_coords,
        y0=y0,
        x0=x0,
        orientation=orientation,
        minor_axis_length=float(minor_axis_length),
        length=float(length_px),
    )
    if not np.isfinite(width_px):
        return _failed_result()

    length = float(length_px) * pixel_size
    width = float(width_px) * pixel_size
    volume = mm3_capsule_volume(length, width)
    surface_area = mm3_capsule_surface_area(length, width)
    return {
        "length_px_mm3_feret": float(length_px),
        "width_px_mm3_feret": float(width_px),
        "length_mm3_feret": length,
        "width_mm3_feret": width,
        "volume_mm3_feret": volume,
        "surface_area_mm3_feret": surface_area,
        "surface_area_to_volume_ratio_mm3_feret": surface_area / volume if volume else np.nan,
        "method_mm3_feret": "mm3_feret",
    }


def _boundary_coordinates(cropped_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    padded = np.pad(cropped_mask, 1, "constant")
    distance_image = ndi.distance_transform_edt(padded)
    rows, cols = np.where(distance_image == 1)
    return np.column_stack((rows, cols)).astype(float)


def _mm3_axis_orientation(orientation: float) -> float:
    if orientation > 0:
        axis_orientation = -math.pi / 2.0 + orientation
    else:
        axis_orientation = math.pi / 2.0 + orientation
    return axis_orientation


def _mm3_orientation_components(orientation: float) -> tuple[float, float, float]:
    axis_orientation = _mm3_axis_orientation(orientation)
    return math.cos(axis_orientation), math.sin(axis_orientation), axis_orientation


def _mm3_length(
    boundary_coords: NDArray[np.float64],
    *,
    y0: float,
    x0: float,
    orientation: float,
    major_axis_length: float,
) -> float:
    cosorient, sinorient, axis_orientation = _mm3_orientation_components(orientation)
    split = int(np.round(boundary_coords.shape[0] / 4.0))
    if axis_orientation > 0:
        first_coords = boundary_coords[:split]
        second_coords = boundary_coords[split:]
    else:
        first_coords = boundary_coords[split:]
        second_coords = boundary_coords[:split]
    if first_coords.shape[0] == 0 or second_coords.shape[0] == 0:
        return np.nan

    amp_param = 1.2
    first_pole = np.array(
        [
            y0 - sinorient * 0.5 * major_axis_length * amp_param,
            x0 + cosorient * 0.5 * major_axis_length * amp_param,
        ],
        dtype=float,
    )
    second_pole = np.array(
        [
            y0 + sinorient * 0.5 * major_axis_length * amp_param,
            x0 - cosorient * 0.5 * major_axis_length * amp_param,
        ],
        dtype=float,
    )
    first_point = _closest_coordinate(first_coords, first_pole)
    second_point = _closest_coordinate(second_coords, second_pole)
    return float(np.linalg.norm(first_point - second_point))


def _mm3_width(
    boundary_coords: NDArray[np.float64],
    *,
    y0: float,
    x0: float,
    orientation: float,
    minor_axis_length: float,
    length: float,
) -> float:
    cosorient, sinorient, axis_orientation = _mm3_orientation_components(orientation)
    split = int(np.round(boundary_coords.shape[0] / 2.0))
    if axis_orientation > 0:
        width_coord_sets = (boundary_coords[:split], boundary_coords[split:])
    else:
        width_coord_sets = (boundary_coords[split:], boundary_coords[:split])
    if width_coord_sets[0].shape[0] == 0 or width_coord_sets[1].shape[0] == 0:
        return np.nan

    amp_param = 1.2
    x1 = x0 + cosorient * 0.5 * length * 0.4
    y1 = y0 - sinorient * 0.5 * length * 0.4
    x2 = x0 - cosorient * 0.5 * length * 0.4
    y2 = y0 + sinorient * 0.5 * length * 0.4
    first_side_points = np.array(
        [
            [
                y1 - cosorient * 0.5 * minor_axis_length * amp_param,
                x1 - sinorient * 0.5 * minor_axis_length * amp_param,
            ],
            [
                y2 - cosorient * 0.5 * minor_axis_length * amp_param,
                x2 - sinorient * 0.5 * minor_axis_length * amp_param,
            ],
        ],
        dtype=float,
    )
    second_side_points = np.array(
        [
            [
                y1 + cosorient * 0.5 * minor_axis_length * amp_param,
                x1 + sinorient * 0.5 * minor_axis_length * amp_param,
            ],
            [
                y2 + cosorient * 0.5 * minor_axis_length * amp_param,
                x2 + sinorient * 0.5 * minor_axis_length * amp_param,
            ],
        ],
        dtype=float,
    )

    widths = []
    for index, (first_point, second_point) in enumerate(
        zip(first_side_points, second_side_points)
    ):
        first_edge = _closest_coordinate(width_coord_sets[index], first_point)
        second_edge = _closest_coordinate(width_coord_sets[index], second_point)
        widths.append(float(np.linalg.norm(first_edge - second_edge)))
    return float(np.mean(widths))


def _closest_coordinate(
    coordinates: NDArray[np.float64],
    point: NDArray[np.float64],
) -> NDArray[np.float64]:
    distances = np.sum((coordinates - point) ** 2, axis=1)
    return coordinates[int(np.argmin(distances))]


def _failed_result() -> dict[str, object]:
    return {
        "length_px_mm3_feret": np.nan,
        "width_px_mm3_feret": np.nan,
        "length_mm3_feret": np.nan,
        "width_mm3_feret": np.nan,
        "volume_mm3_feret": np.nan,
        "surface_area_mm3_feret": np.nan,
        "surface_area_to_volume_ratio_mm3_feret": np.nan,
        "method_mm3_feret": "mm3_feret_failed",
    }
