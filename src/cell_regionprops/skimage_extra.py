"""Build extra property functions for `skimage.measure.regionprops_table`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import SupportsFloat, cast

import numpy as np
from numpy.typing import NDArray

from cell_regionprops.morphometrics_mesh import measure_cell_mesh
from cell_regionprops.registry import property_moments_axis

SkimageExtraProperty = Callable[[NDArray[np.bool_]], object]


def make_extra_properties(
    names: Iterable[str],
    *,
    pixel_size: float = 1.0,
) -> tuple[SkimageExtraProperty, ...]:
    """Create named property callables for scikit-image `regionprops_table`.

    Args:
        names: Property names to expose. Supported names are
            `moments_axis_length`, `moments_axis_width`,
            `morphometrics_length`, and `morphometrics_width`.
        pixel_size: Pixel size in physical units. Defaults to 1.0.

    Returns:
        A tuple of named callables suitable for scikit-image's
        `extra_properties` argument.

    Raises:
        KeyError: If a requested property name is unsupported.
    """
    factories = {
        "moments_axis_length": _make_moments_axis_length,
        "moments_axis_width": _make_moments_axis_width,
        "morphometrics_length": _make_morphometrics_length,
        "morphometrics_width": _make_morphometrics_width,
    }

    properties: list[SkimageExtraProperty] = []
    for name in names:
        try:
            factory = factories[name]
        except KeyError as error:
            raise KeyError(f"Unknown extra property: {name}") from error
        properties.append(factory(pixel_size))
    return tuple(properties)


def _make_moments_axis_length(pixel_size: float) -> SkimageExtraProperty:
    def moments_axis_length(mask: NDArray[np.bool_]) -> float:
        result = property_moments_axis(mask, label=1, pixel_size=pixel_size)
        return float(cast(SupportsFloat, result["length_moments"]))

    return moments_axis_length


def _make_moments_axis_width(pixel_size: float) -> SkimageExtraProperty:
    def moments_axis_width(mask: NDArray[np.bool_]) -> float:
        result = property_moments_axis(mask, label=1, pixel_size=pixel_size)
        return float(cast(SupportsFloat, result["width_moments"]))

    return moments_axis_width


def _make_morphometrics_length(pixel_size: float) -> SkimageExtraProperty:
    def morphometrics_length(mask: NDArray[np.bool_]) -> float:
        result = measure_cell_mesh(mask, pixel_size_um=pixel_size)
        return float(cast(SupportsFloat, result["length_um"]))

    return morphometrics_length


def _make_morphometrics_width(pixel_size: float) -> SkimageExtraProperty:
    def morphometrics_width(mask: NDArray[np.bool_]) -> float:
        result = measure_cell_mesh(mask, pixel_size_um=pixel_size)
        return float(cast(SupportsFloat, result["width_um"]))

    return morphometrics_width
