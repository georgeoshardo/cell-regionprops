"""Tabulate cell measurements from binary and labeled masks."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from cell_regionprops.registry import DEFAULT_PROPERTIES, PROPERTY_REGISTRY

ExtraProperty: TypeAlias = Callable[[NDArray[np.bool_]], object]


def _normalise_properties(properties: Iterable[str] | None) -> tuple[str, ...]:
    if properties is None:
        return DEFAULT_PROPERTIES
    return tuple(properties)


def _labels_to_measure(
    label_image: NDArray[np.integer],
    labels: Iterable[int] | None,
) -> list[int]:
    if labels is not None:
        return [int(label) for label in labels]
    unique_labels = np.unique(label_image)
    return [int(label) for label in unique_labels if int(label) != 0]


def _validate_label_image(label_image: ArrayLike) -> NDArray[np.integer]:
    image = np.asarray(label_image)
    if image.ndim != 2:
        raise ValueError("`label_image` must be a 2D array.")
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError("`label_image` must contain integer labels.")
    return image


def _extra_property_row(
    mask: NDArray[np.bool_],
    extra_properties: Mapping[str, ExtraProperty] | None,
) -> dict[str, object]:
    if extra_properties is None:
        return {}
    return {
        name: function(mask)
        for name, function in extra_properties.items()
    }


def regionprops_table(
    label_image: ArrayLike,
    *,
    pixel_size: float = 1.0,
    labels: Iterable[int] | None = None,
    properties: Iterable[str] | None = None,
    extra_properties: Mapping[str, ExtraProperty] | None = None,
) -> pd.DataFrame:
    """Measure requested properties for each label in a 2D mask.

    Args:
        label_image: Two-dimensional integer label image. Label 0 is treated as
            background and is skipped unless requested explicitly through
            `labels`.
        pixel_size: Size of one pixel in physical units. Defaults to 1.0.
        labels: Optional label sequence to measure. Missing requested labels are
            returned as empty masks, so the requested label still appears in the
            output.
        properties: Names of built-in properties to compute. Defaults to all
            built-in properties.
        extra_properties: Additional named functions that receive each binary
            object mask and return one scalar-like value.

    Returns:
        A dataframe with one row per measured label.

    Raises:
        KeyError: If a requested built-in property is not registered.
        TypeError: If `label_image` is not integer labeled.
        ValueError: If `label_image` is not two-dimensional.
    """
    image = _validate_label_image(label_image)
    selected_properties = _normalise_properties(properties)
    measured_labels = _labels_to_measure(image, labels)

    rows: list[dict[str, object]] = []
    for label in measured_labels:
        mask = image == label
        row: dict[str, object] = {}
        for property_name in selected_properties:
            try:
                property_function = PROPERTY_REGISTRY[property_name]
            except KeyError as error:
                raise KeyError(f"Unknown property: {property_name}") from error
            row.update(property_function(mask, label, pixel_size))
        row.update(_extra_property_row(mask, extra_properties))
        rows.append(row)

    return pd.DataFrame(rows)


def binary_regionprops_table(
    binary_mask: ArrayLike,
    *,
    pixel_size: float = 1.0,
    properties: Iterable[str] | None = None,
    extra_properties: Mapping[str, ExtraProperty] | None = None,
) -> pd.DataFrame:
    """Measure properties for a single foreground object in a binary mask.

    Args:
        binary_mask: Two-dimensional array where nonzero values are foreground.
        pixel_size: Size of one pixel in physical units. Defaults to 1.0.
        properties: Names of built-in properties to compute. Defaults to all
            built-in properties.
        extra_properties: Additional named functions that receive the binary
            foreground mask and return one scalar-like value.

    Returns:
        A dataframe containing a single row for label 1.

    Raises:
        ValueError: If `binary_mask` is not two-dimensional.
    """
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("`binary_mask` must be a 2D array.")

    label_image = mask.astype(np.uint8)
    return regionprops_table(
        label_image,
        pixel_size=pixel_size,
        labels=(1,),
        properties=properties,
        extra_properties=extra_properties,
    )


def stack_regionprops_table(
    label_stack: ArrayLike,
    *,
    index_names: Sequence[str] | None = None,
    pixel_size: float = 1.0,
    properties: Iterable[str] | None = None,
    extra_properties: Mapping[str, ExtraProperty] | None = None,
) -> pd.DataFrame:
    """Measure labels in every 2D frame of a stack.

    The last two axes are interpreted as `y, x`; all leading axes are iterated
    and copied into index columns. Empty frames contribute no rows unless
    labels are requested by calling `regionprops_table` directly.

    Args:
        label_stack: Integer labeled array with shape `(..., y, x)`.
        index_names: Optional names for the leading stack axes. Defaults to
            `axis_0`, `axis_1`, and so on.
        pixel_size: Size of one pixel in physical units. Defaults to 1.0.
        properties: Names of built-in properties to compute. Defaults to all
            built-in properties.
        extra_properties: Additional named functions that receive each binary
            object mask and return one scalar-like value.

    Returns:
        A dataframe containing index columns followed by measurement columns.

    Raises:
        TypeError: If `label_stack` is not integer labeled.
        ValueError: If `label_stack` has fewer than three dimensions or the
            number of `index_names` does not match the leading axes.
    """
    stack = np.asarray(label_stack)
    if stack.ndim < 3:
        raise ValueError("`label_stack` must have at least three dimensions.")
    if not np.issubdtype(stack.dtype, np.integer):
        raise TypeError("`label_stack` must contain integer labels.")

    leading_shape = stack.shape[:-2]
    names = tuple(index_names) if index_names is not None else tuple(
        f"axis_{axis}" for axis in range(len(leading_shape))
    )
    if len(names) != len(leading_shape):
        raise ValueError("`index_names` must match the number of leading axes.")

    tables: list[pd.DataFrame] = []
    for index in np.ndindex(leading_shape):
        frame = stack[index]
        table = regionprops_table(
            frame,
            pixel_size=pixel_size,
            properties=properties,
            extra_properties=extra_properties,
        )
        if table.empty:
            continue
        for name, value in reversed(list(zip(names, index))):
            table.insert(0, name, value)
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=list(names))
    return pd.concat(tables, ignore_index=True)
