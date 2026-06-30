"""Tabulate cell measurements from binary and labeled masks."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, TypeAlias

import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from cell_regionprops.registry import DEFAULT_PROPERTIES, PROPERTY_REGISTRY

ExtraProperty: TypeAlias = Callable[[NDArray[np.bool_]], object]
ExecutionMode: TypeAlias = Literal["auto", "vectorized", "loop"]

_VECTORIZABLE_PROPERTIES = {"label", "area", "centroid", "moments_axis"}


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


def _is_dask_array(value: object) -> bool:
    return isinstance(value, da.Array)


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
    label_stack: object,
    *,
    index_names: Sequence[str] | None = None,
    pixel_size: float = 1.0,
    properties: Iterable[str] | None = None,
    extra_properties: Mapping[str, ExtraProperty] | None = None,
    execution: ExecutionMode = "auto",
) -> pd.DataFrame:
    """Measure labels in every 2D frame of a stack.

    The last two axes are interpreted as `y, x`; all leading axes are iterated
    and copied into index columns. Empty frames contribute no rows unless
    labels are requested by calling `regionprops_table` directly. Area,
    centroid, and moment-axis properties are computed with a vectorized stack
    path by default; non-vectorizable properties use the loop path.

    Args:
        label_stack: Integer labeled array-like object with shape
            `(..., y, x)`. NumPy and Dask arrays are supported.
        index_names: Optional names for the leading stack axes. Defaults to
            `axis_0`, `axis_1`, and so on.
        pixel_size: Size of one pixel in physical units. Defaults to 1.0.
        properties: Names of built-in properties to compute. Defaults to all
            built-in properties.
        extra_properties: Additional named functions that receive each binary
            object mask and return one scalar-like value.
        execution: Execution mode. `"auto"` uses the vectorized path when all
            requested properties support it, `"vectorized"` requires that path,
            and `"loop"` forces the single-frame loop. Defaults to `"auto"`.

    Returns:
        A dataframe containing index columns followed by measurement columns.

    Raises:
        TypeError: If `label_stack` is not integer labeled.
        ValueError: If `label_stack` has fewer than three dimensions or the
            number of `index_names` does not match the leading axes, or if
            `"vectorized"` is requested for unsupported properties.
    """
    stack: Any = label_stack if _is_dask_array(label_stack) else np.asarray(label_stack)
    if stack.ndim < 3:
        raise ValueError("`label_stack` must have at least three dimensions.")
    if not np.issubdtype(stack.dtype, np.integer):
        raise TypeError("`label_stack` must contain integer labels.")
    if execution not in {"auto", "vectorized", "loop"}:
        raise ValueError("`execution` must be one of 'auto', 'vectorized', or 'loop'.")

    leading_shape = stack.shape[:-2]
    names = tuple(index_names) if index_names is not None else tuple(
        f"axis_{axis}" for axis in range(len(leading_shape))
    )
    if len(names) != len(leading_shape):
        raise ValueError("`index_names` must match the number of leading axes.")

    selected_properties = _normalise_properties(properties)
    supports_vectorized = (
        extra_properties is None
        and set(selected_properties).issubset(_VECTORIZABLE_PROPERTIES)
    )
    if execution == "vectorized" and not supports_vectorized:
        raise ValueError("Requested properties are not supported by the vectorized stack path.")
    if execution == "vectorized" or (execution == "auto" and supports_vectorized):
        return _stack_regionprops_table_vectorized(
            stack,
            index_names=names,
            pixel_size=pixel_size,
            properties=selected_properties,
        )
    return _stack_regionprops_table_loop(
        np.asarray(stack),
        index_names=names,
        pixel_size=pixel_size,
        properties=selected_properties,
        extra_properties=extra_properties,
    )


def _stack_regionprops_table_loop(
    stack: NDArray[np.integer],
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
    extra_properties: Mapping[str, ExtraProperty] | None,
) -> pd.DataFrame:
    leading_shape = stack.shape[:-2]
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
        for name, value in reversed(list(zip(index_names, index))):
            table.insert(0, name, value)
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=list(index_names))
    return pd.concat(tables, ignore_index=True)


def _stack_regionprops_table_vectorized(
    stack: Any,
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
) -> pd.DataFrame:
    leading_shape = stack.shape[:-2]
    height, width = stack.shape[-2:]
    frames = stack.reshape((-1, height, width))
    if _is_dask_array(stack):
        unique_labels = da.unique(stack).compute()
    else:
        unique_labels = np.unique(stack)
    labels = [int(label) for label in unique_labels if int(label) != 0]
    if not labels:
        return pd.DataFrame(columns=list(index_names))

    y = np.arange(height, dtype=float)[:, None]
    x = np.arange(width, dtype=float)[None, :]

    rows: list[dict[str, object]] = []
    for label in labels:
        mask = frames == label
        area = mask.sum(axis=(1, 2))
        safe_area = da.where(area > 0, area, np.nan) if _is_dask_array(area) else np.where(
            area > 0,
            area,
            np.nan,
        )
        x_mean = (mask * x).sum(axis=(1, 2)) / safe_area
        y_mean = (mask * y).sum(axis=(1, 2)) / safe_area
        mu20 = (mask * x**2).sum(axis=(1, 2)) / safe_area - x_mean**2
        mu02 = (mask * y**2).sum(axis=(1, 2)) / safe_area - y_mean**2
        mu11 = (mask * x * y).sum(axis=(1, 2)) / safe_area - x_mean * y_mean
        if _is_dask_array(mask):
            eigenvalue_gap = da.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11**2)
            major_eigenvalue = da.maximum((mu20 + mu02 + eigenvalue_gap) / 2.0, 0.0)
            minor_eigenvalue = da.maximum((mu20 + mu02 - eigenvalue_gap) / 2.0, 0.0)
            length_px = 4.0 * da.sqrt(major_eigenvalue)
            width_px = 4.0 * da.sqrt(minor_eigenvalue)
            area_np, x_mean_np, y_mean_np, length_px_np, width_px_np = da.compute(
                area,
                x_mean,
                y_mean,
                length_px,
                width_px,
            )
        else:
            eigenvalue_gap = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11**2)
            major_eigenvalue = np.maximum((mu20 + mu02 + eigenvalue_gap) / 2.0, 0.0)
            minor_eigenvalue = np.maximum((mu20 + mu02 - eigenvalue_gap) / 2.0, 0.0)
            length_px_np = 4.0 * np.sqrt(major_eigenvalue)
            width_px_np = 4.0 * np.sqrt(minor_eigenvalue)
            area_np = np.asarray(area)
            x_mean_np = np.asarray(x_mean)
            y_mean_np = np.asarray(y_mean)

        present_frame_indices = np.flatnonzero(area_np > 0)
        if present_frame_indices.size == 0:
            continue

        unraveled = np.unravel_index(present_frame_indices, leading_shape)
        for row_offset, flat_index in enumerate(present_frame_indices):
            row: dict[str, object] = {
                name: int(unraveled[axis][row_offset])
                for axis, name in enumerate(index_names)
            }
            for property_name in properties:
                if property_name == "label":
                    row["label"] = label
                elif property_name == "area":
                    row["area_px"] = int(area_np[flat_index])
                    row["area"] = int(area_np[flat_index]) * pixel_size**2
                elif property_name == "centroid":
                    row["centroid_y"] = float(y_mean_np[flat_index])
                    row["centroid_x"] = float(x_mean_np[flat_index])
                elif property_name == "moments_axis":
                    row["length_px_moments"] = float(length_px_np[flat_index])
                    row["width_px_moments"] = float(width_px_np[flat_index])
                    row["length_moments"] = float(length_px_np[flat_index] * pixel_size)
                    row["width_moments"] = float(width_px_np[flat_index] * pixel_size)
                else:
                    raise KeyError(f"Unknown vectorized property: {property_name}")
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=list(index_names))
    sort_columns = [*index_names]
    if "label" in table.columns:
        sort_columns.append("label")
    return table.sort_values(sort_columns).reset_index(drop=True)
