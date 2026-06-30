"""Morphometrics-style properties for labeled cell masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cell_regionprops.morphometrics_mesh import measure_cell_mesh


def property_morphometrics(
    mask: NDArray[np.bool_],
    label: int,
    pixel_size: float,
) -> dict[str, object]:
    """Measure contour/Voronoi rib-mesh geometry for one object mask.

    Args:
        mask: Binary mask for one labeled object.
        label: Integer label represented by `mask`.
        pixel_size: Pixel size in microns.

    Returns:
        A dictionary containing Morphometrics-style geometry columns with a
        `_morphometrics` suffix.
    """
    del label
    measurement = measure_cell_mesh(mask, pixel_size_um=pixel_size)
    return {
        "length_px_morphometrics": measurement["length_px"],
        "width_px_morphometrics": measurement["width_px"],
        "length_morphometrics": measurement["length_um"],
        "width_morphometrics": measurement["width_um"],
        "volume_morphometrics": measurement["volume_um3"],
        "surface_area_morphometrics": measurement["surface_area_um2"],
        "surface_area_to_volume_ratio_morphometrics": measurement[
            "surface_area_to_volume_um_inv"
        ],
        "method_morphometrics": measurement["method"],
        "centerline_xy_morphometrics": measurement["centerline_xy"],
        "width_profile_px_morphometrics": measurement["width_profile_px"],
        "mesh_px_morphometrics": measurement["mesh_px"],
    }
