"""Tabulate cell measurements from binary and labeled masks."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, TypeAlias

import dask.array as da
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from cell_regionprops.registry import DEFAULT_PROPERTIES, PROPERTY_REGISTRY

ExtraProperty: TypeAlias = Callable[[NDArray[np.bool_]], object]
MomentsBackend: TypeAlias = Literal["label_loop", "numba"]

_VECTORIZABLE_PROPERTIES = {"label", "area", "centroid", "moments_axis"}
_NUMBA_DIRECT_LABEL_MAX_BINS = 10_000_000


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
    moments_backend: MomentsBackend = "label_loop",
    moments_chunk_size: int | None = None,
    morphometrics_n_jobs: int = -1,
) -> pd.DataFrame:
    """Measure labels in every 2D frame of a stack.

    The last two axes are interpreted as `y, x`; all leading axes are iterated
    and copied into index columns. Empty frames contribute no rows unless
    labels are requested by calling `regionprops_table` directly. Label, area,
    centroid, and moment-axis properties are always computed with the vectorized
    stack path. Morphometrics and extra properties are computed per object and
    merged back onto the vectorized results.

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
        moments_backend: Vectorized backend for label, area, centroid, and
            moment-axis measurements. `"label_loop"` loops over label values;
            `"numba"` uses a parallel compiled loop over 2D frames. Defaults
            to `"label_loop"`.
        moments_chunk_size: Optional number of first-axis slices per Numba
            chunk. Dask arrays using the Numba backend are chunked by default
            along the first leading axis; NumPy arrays are processed as one
            batch unless this is set.
        morphometrics_n_jobs: Number of parallel workers for non-vectorized
            morphometrics and extra-property measurements. `-1` uses all cores;
            `0` and `1` run serially. Defaults to `-1`.

    Returns:
        A dataframe containing index columns followed by measurement columns.

    Raises:
        TypeError: If `label_stack` is not integer labeled.
        ValueError: If `label_stack` has fewer than three dimensions or the
            number of `index_names` does not match the leading axes.
    """
    stack: Any = label_stack if _is_dask_array(label_stack) else np.asarray(label_stack)
    if stack.ndim < 3:
        raise ValueError("`label_stack` must have at least three dimensions.")
    if not np.issubdtype(stack.dtype, np.integer):
        raise TypeError("`label_stack` must contain integer labels.")
    if moments_backend not in {"label_loop", "numba"}:
        raise ValueError("`moments_backend` must be 'label_loop' or 'numba'.")
    if moments_chunk_size is not None and moments_chunk_size < 1:
        raise ValueError("`moments_chunk_size` must be a positive integer or None.")

    leading_shape = stack.shape[:-2]
    names = tuple(index_names) if index_names is not None else tuple(
        f"axis_{axis}" for axis in range(len(leading_shape))
    )
    if len(names) != len(leading_shape):
        raise ValueError("`index_names` must match the number of leading axes.")

    selected_properties = _normalise_properties(properties)
    vectorized_properties = tuple(
        property_name
        for property_name in selected_properties
        if property_name in _VECTORIZABLE_PROPERTIES
    )
    per_object_properties = tuple(
        property_name
        for property_name in selected_properties
        if property_name not in _VECTORIZABLE_PROPERTIES
    )
    vectorized_properties_for_compute = (
        ("label", *vectorized_properties)
        if (per_object_properties or extra_properties is not None)
        and "label" not in vectorized_properties
        else vectorized_properties
    )

    vectorized_table = (
        _stack_regionprops_table_vectorized(
            stack,
            index_names=names,
            pixel_size=pixel_size,
            properties=vectorized_properties_for_compute,
            moments_backend=moments_backend,
            moments_chunk_size=moments_chunk_size,
        )
        if vectorized_properties
        else pd.DataFrame()
    )
    per_object_table = (
        _stack_regionprops_table_per_object(
            _compute_stack_if_needed(stack),
            index_names=names,
            pixel_size=pixel_size,
            properties=per_object_properties,
            extra_properties=extra_properties,
            n_jobs=morphometrics_n_jobs,
        )
        if per_object_properties or extra_properties is not None
        else pd.DataFrame()
    )
    return _merge_stack_property_tables(
        vectorized_table,
        per_object_table,
        index_names=names,
    )


def _compute_stack_if_needed(stack: Any) -> NDArray[np.integer]:
    if _is_dask_array(stack):
        return np.asarray(stack.compute())
    return np.asarray(stack)


def _normalise_n_jobs(n_jobs: int) -> int:
    return 1 if n_jobs == 0 else n_jobs


def _stack_regionprops_table_per_object(
    stack: NDArray[np.integer],
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
    extra_properties: Mapping[str, ExtraProperty] | None,
    n_jobs: int,
) -> pd.DataFrame:
    leading_shape = stack.shape[:-2]
    frame_indices = list(np.ndindex(leading_shape))
    requested_properties = ("label", *properties)

    def measure_frame(index: tuple[int, ...]) -> list[dict[str, object]]:
        frame = stack[index]
        table = regionprops_table(
            frame,
            pixel_size=pixel_size,
            properties=requested_properties,
            extra_properties=extra_properties,
        )
        if table.empty:
            return []
        prefix = {
            name: int(value)
            for name, value in zip(index_names, index)
        }
        return [
            {**prefix, **row}
            for row in table.to_dict("records")
        ]

    normalised_n_jobs = _normalise_n_jobs(n_jobs)
    progress = tqdm(
        frame_indices,
        desc="Morphometrics frames",
        unit="frame",
        leave=False,
    )
    if normalised_n_jobs == 1:
        measured_rows = [measure_frame(index) for index in progress]
    else:
        measured_rows = Parallel(n_jobs=normalised_n_jobs)(
            delayed(measure_frame)(index)
            for index in progress
        )

    rows = [
        row
        for frame_rows in measured_rows
        for row in frame_rows
    ]

    if not rows:
        return pd.DataFrame(columns=list(index_names))
    return pd.DataFrame(rows)


def _merge_stack_property_tables(
    vectorized_table: pd.DataFrame,
    per_object_table: pd.DataFrame,
    *,
    index_names: tuple[str, ...],
) -> pd.DataFrame:
    if vectorized_table.empty:
        return per_object_table
    if per_object_table.empty:
        return vectorized_table
    merge_columns = [*index_names, "label"]
    return vectorized_table.merge(
        per_object_table,
        on=merge_columns,
        how="outer",
        sort=True,
    )


def _stack_regionprops_table_vectorized(
    stack: Any,
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
    moments_backend: MomentsBackend,
    moments_chunk_size: int | None,
) -> pd.DataFrame:
    if moments_backend == "numba":
        if moments_chunk_size is not None or _is_dask_array(stack):
            chunk_size = moments_chunk_size or min(int(stack.shape[0]), 64)
            return _stack_regionprops_table_numba_chunked(
                stack,
                index_names=index_names,
                pixel_size=pixel_size,
                properties=properties,
                chunk_size=chunk_size,
            )
        return _stack_regionprops_table_numba(
            np.asarray(stack),
            index_names=index_names,
            pixel_size=pixel_size,
            properties=properties,
        )
    return _stack_regionprops_table_label_loop(
        stack,
        index_names=index_names,
        pixel_size=pixel_size,
        properties=properties,
    )


def _stack_regionprops_table_label_loop(
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


@njit(parallel=True, cache=False)
def _numba_label_moments_kernel(
    frames: NDArray[np.int32],
    max_label: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    n_frames, height, width = frames.shape
    area = np.zeros((n_frames, max_label + 1), dtype=np.int64)
    sum_x = np.zeros((n_frames, max_label + 1), dtype=np.float64)
    sum_y = np.zeros((n_frames, max_label + 1), dtype=np.float64)
    sum_x2 = np.zeros((n_frames, max_label + 1), dtype=np.float64)
    sum_y2 = np.zeros((n_frames, max_label + 1), dtype=np.float64)
    sum_xy = np.zeros((n_frames, max_label + 1), dtype=np.float64)

    for frame_index in prange(n_frames):
        for y in range(height):
            y_float = float(y)
            for x in range(width):
                label = frames[frame_index, y, x]
                if label == 0:
                    continue
                x_float = float(x)
                area[frame_index, label] += 1
                sum_x[frame_index, label] += x_float
                sum_y[frame_index, label] += y_float
                sum_x2[frame_index, label] += x_float * x_float
                sum_y2[frame_index, label] += y_float * y_float
                sum_xy[frame_index, label] += x_float * y_float

    return area, sum_x, sum_y, sum_x2, sum_y2, sum_xy


def _numba_moments_columns_to_dataframe(
    columns: dict[str, NDArray[Any]],
    *,
    index_names: tuple[str, ...],
) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=list(index_names))

    table = pd.DataFrame(columns)
    sort_columns = [*index_names]
    if "label" in table.columns:
        sort_columns.append("label")
    return table.sort_values(sort_columns).reset_index(drop=True)


def _concatenate_column_chunks(
    column_chunks: list[dict[str, NDArray[Any]]],
) -> dict[str, NDArray[Any]]:
    if not column_chunks:
        return {}
    column_names = tuple(column_chunks[0])
    return {
        column_name: np.concatenate(
            [chunk[column_name] for chunk in column_chunks],
        )
        for column_name in column_names
    }


def _stack_regionprops_table_numba_chunked(
    stack: Any,
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
    chunk_size: int,
) -> pd.DataFrame:
    column_chunks: list[dict[str, NDArray[Any]]] = []
    chunk_starts = range(0, int(stack.shape[0]), chunk_size)
    for start in tqdm(
        chunk_starts,
        desc="Moments chunks",
        unit="chunk",
        leave=False,
    ):
        stop = min(start + chunk_size, int(stack.shape[0]))
        chunk = _compute_stack_if_needed(stack[start:stop])
        columns = _stack_regionprops_table_numba_columns(
            chunk,
            index_names=index_names,
            pixel_size=pixel_size,
            properties=properties,
        )
        if not columns:
            continue
        columns[index_names[0]] = columns[index_names[0]] + start
        column_chunks.append(columns)

    return _numba_moments_columns_to_dataframe(
        _concatenate_column_chunks(column_chunks),
        index_names=index_names,
    )


def _stack_regionprops_table_numba(
    stack: NDArray[np.integer],
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
) -> pd.DataFrame:
    return _numba_moments_columns_to_dataframe(
        _stack_regionprops_table_numba_columns(
            stack,
            index_names=index_names,
            pixel_size=pixel_size,
            properties=properties,
        ),
        index_names=index_names,
    )


def _stack_regionprops_table_numba_columns(
    stack: NDArray[np.integer],
    *,
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
) -> dict[str, NDArray[Any]]:
    stack_np = np.asarray(stack)
    leading_shape = stack_np.shape[:-2]
    height, width = stack_np.shape[-2:]
    n_frames = int(np.prod(leading_shape))
    if np.any(stack_np < 0):
        raise ValueError("The Numba moments backend requires non-negative labels.")

    foreground = stack_np != 0
    if not np.any(foreground):
        return {}

    max_label = int(stack_np.max())
    dense_bin_count = n_frames * (max_label + 1)
    if dense_bin_count <= _NUMBA_DIRECT_LABEL_MAX_BINS:
        frames = stack_np.astype(np.int32, copy=False).reshape((n_frames, height, width))
        label_by_code = np.arange(max_label + 1, dtype=np.int64)
        area_matrix, sum_x_matrix, sum_y_matrix, sum_x2_matrix, sum_y2_matrix, sum_xy_matrix = (
            _numba_label_moments_kernel(frames, max_label)
        )
    else:
        foreground_labels = stack_np[foreground]
        labels, label_codes = np.unique(foreground_labels, return_inverse=True)
        remapped = np.zeros(stack_np.shape, dtype=np.int32)
        remapped[foreground] = label_codes.astype(np.int32) + 1
        frames = remapped.reshape((n_frames, height, width))
        label_by_code = np.zeros(labels.size + 1, dtype=np.int64)
        label_by_code[1:] = labels.astype(np.int64)
        area_matrix, sum_x_matrix, sum_y_matrix, sum_x2_matrix, sum_y2_matrix, sum_xy_matrix = (
            _numba_label_moments_kernel(frames, int(labels.size))
        )

    return _numba_moments_columns_from_matrices(
        area_matrix=area_matrix,
        sum_x_matrix=sum_x_matrix,
        sum_y_matrix=sum_y_matrix,
        sum_x2_matrix=sum_x2_matrix,
        sum_y2_matrix=sum_y2_matrix,
        sum_xy_matrix=sum_xy_matrix,
        label_by_code=label_by_code,
        leading_shape=leading_shape,
        index_names=index_names,
        pixel_size=pixel_size,
        properties=properties,
    )


def _numba_moments_columns_from_matrices(
    *,
    area_matrix: NDArray[np.int64],
    sum_x_matrix: NDArray[np.float64],
    sum_y_matrix: NDArray[np.float64],
    sum_x2_matrix: NDArray[np.float64],
    sum_y2_matrix: NDArray[np.float64],
    sum_xy_matrix: NDArray[np.float64],
    label_by_code: NDArray[np.int64],
    leading_shape: tuple[int, ...],
    index_names: tuple[str, ...],
    pixel_size: float,
    properties: tuple[str, ...],
) -> dict[str, NDArray[Any]]:
    present_frame_indices, present_label_codes = np.nonzero(area_matrix[:, 1:])
    present_label_codes = present_label_codes + 1
    if present_frame_indices.size == 0:
        return {}

    unraveled = np.unravel_index(present_frame_indices, leading_shape)
    area_px = area_matrix[present_frame_indices, present_label_codes]
    x_mean = sum_x_matrix[present_frame_indices, present_label_codes] / area_px
    y_mean = sum_y_matrix[present_frame_indices, present_label_codes] / area_px
    mu20 = sum_x2_matrix[present_frame_indices, present_label_codes] / area_px - x_mean**2
    mu02 = sum_y2_matrix[present_frame_indices, present_label_codes] / area_px - y_mean**2
    mu11 = sum_xy_matrix[present_frame_indices, present_label_codes] / area_px - x_mean * y_mean
    eigenvalue_gap = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11**2)
    major_eigenvalue = np.maximum((mu20 + mu02 + eigenvalue_gap) / 2.0, 0.0)
    minor_eigenvalue = np.maximum((mu20 + mu02 - eigenvalue_gap) / 2.0, 0.0)
    length_px = 4.0 * np.sqrt(major_eigenvalue)
    width_px = 4.0 * np.sqrt(minor_eigenvalue)

    columns: dict[str, NDArray[Any]] = {
        name: np.asarray(unraveled[axis], dtype=np.int64)
        for axis, name in enumerate(index_names)
    }
    for property_name in properties:
        if property_name == "label":
            columns["label"] = label_by_code[present_label_codes]
        elif property_name == "area":
            columns["area_px"] = area_px
            columns["area"] = area_px.astype(float) * pixel_size**2
        elif property_name == "centroid":
            columns["centroid_y"] = y_mean
            columns["centroid_x"] = x_mean
        elif property_name == "moments_axis":
            columns["length_px_moments"] = length_px
            columns["width_px_moments"] = width_px
            columns["length_moments"] = length_px * pixel_size
            columns["width_moments"] = width_px * pixel_size
        else:
            raise KeyError(f"Unknown numba property: {property_name}")
    return columns
