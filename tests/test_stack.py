import numpy as np
import dask.array as da
import pandas as pd
import pytest
from typing import Any

from cell_regionprops import regionprops_table
from cell_regionprops import stack_regionprops_table
from cell_regionprops import core


def test_stack_regionprops_table_adds_leading_axis_indices() -> None:
    stack = np.zeros((2, 3, 5, 5), dtype=np.uint8)
    stack[0, 0, 1:3, 1:3] = 1
    stack[1, 2, 2:4, 2:4] = 2

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area"),
    )

    assert table[["sample", "frame", "label", "area_px"]].to_dict("records") == [
        {"sample": 0, "frame": 0, "label": 1, "area_px": 4},
        {"sample": 1, "frame": 2, "label": 2, "area_px": 4},
    ]


def test_stack_regionprops_table_rejects_index_name_mismatch() -> None:
    stack = np.zeros((2, 3, 5, 5), dtype=np.uint8)

    try:
        stack_regionprops_table(stack, index_names=("sample",))
    except ValueError as error:
        assert "index_names" in str(error)
    else:
        raise AssertionError("Expected stack_regionprops_table to reject mismatched names.")


def test_stack_regionprops_table_computes_moments_without_execution_argument() -> None:
    stack = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    stack[0, 0, 1:3, 1:3] = 1
    stack[0, 1, 1:4, 1:3] = 1

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
    )

    assert table["frame"].tolist() == [0, 1]
    assert table["label"].tolist() == [1, 1]
    assert table["area_px"].tolist() == [4, 6]
    assert table["length_px_moments"].notna().all()
    assert table["width_px_moments"].notna().all()


def test_stack_regionprops_table_moments_match_single_frame_table() -> None:
    stack = np.zeros((2, 3, 6, 6), dtype=np.uint8)
    stack[0, 0, 1:3, 1:4] = 1
    stack[0, 1, 2:5, 2:4] = 2
    stack[1, 2, 1:5, 1:3] = 1

    stack_table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)

    single_frame_tables = []
    for sample, frame in np.ndindex(stack.shape[:-2]):
        table = regionprops_table(
            stack[sample, frame],
            properties=("label", "area", "centroid", "moments_axis"),
        )
        if table.empty:
            continue
        table.insert(0, "frame", frame)
        table.insert(0, "sample", sample)
        single_frame_tables.append(table)
    expected = pd.concat(single_frame_tables, ignore_index=True)
    expected = expected.sort_values(["sample", "frame", "label"]).reset_index(drop=True)

    assert stack_table.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(
        stack_table.drop(columns=["sample", "frame", "label"]).to_numpy(dtype=float),
        expected.drop(columns=["sample", "frame", "label"]).to_numpy(dtype=float),
    )


def test_stack_regionprops_table_numba_matches_label_loop_for_moments() -> None:
    stack = np.zeros((2, 3, 7, 8), dtype=np.int32)
    stack[0, 0, 1:3, 1:5] = 1
    stack[0, 0, 4:6, 2:7] = 18
    stack[0, 1, 2:6, 3:5] = 1
    stack[1, 0, 1:6, 1:3] = 18
    stack[1, 2, 3:6, 4:7] = 1

    label_loop = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="label_loop",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)
    numba = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="numba",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)

    assert numba.columns.tolist() == label_loop.columns.tolist()
    assert numba[["sample", "frame", "label", "area_px"]].equals(
        label_loop[["sample", "frame", "label", "area_px"]]
    )
    np.testing.assert_allclose(
        numba.drop(columns=["sample", "frame", "label", "area_px"]).to_numpy(dtype=float),
        label_loop.drop(columns=["sample", "frame", "label", "area_px"]).to_numpy(dtype=float),
    )


def test_stack_regionprops_table_numba_uses_small_integer_labels_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = np.zeros((1, 2, 6, 6), dtype=np.uint8)
    stack[0, 0, 1:3, 1:4] = 1
    stack[0, 1, 2:5, 3:5] = 2

    def fail_searchsorted(*args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("Numba backend should not remap small integer labels.")

    monkeypatch.setattr(core.np, "searchsorted", fail_searchsorted)

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="numba",
    )

    assert table[["sample", "frame", "label", "area_px"]].to_dict("records") == [
        {"sample": 0, "frame": 0, "label": 1, "area_px": 6},
        {"sample": 0, "frame": 1, "label": 2, "area_px": 6},
    ]


def test_stack_regionprops_table_numba_chunked_matches_label_loop_for_moments() -> None:
    stack = np.zeros((5, 3, 7, 8), dtype=np.int32)
    stack[0, 0, 1:3, 1:5] = 1
    stack[1, 2, 4:6, 2:7] = 18
    stack[3, 1, 2:6, 3:5] = 1
    stack[4, 0, 1:6, 1:3] = 18

    label_loop = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="label_loop",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)
    numba = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="numba",
        moments_chunk_size=2,
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)

    assert numba[["sample", "frame", "label", "area_px"]].equals(
        label_loop[["sample", "frame", "label", "area_px"]]
    )
    np.testing.assert_allclose(
        numba.drop(columns=["sample", "frame", "label", "area_px"]).to_numpy(dtype=float),
        label_loop.drop(columns=["sample", "frame", "label", "area_px"]).to_numpy(dtype=float),
    )


def test_stack_regionprops_table_numba_merges_with_morphometrics() -> None:
    stack = np.zeros((1, 2, 20, 20), dtype=np.uint8)
    stack[0, 0, 5:15, 7:13] = 1
    stack[0, 1, 4:16, 7:13] = 1

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "moments_axis", "morphometrics"),
        moments_backend="numba",
        morphometrics_n_jobs=0,
    )

    assert table["frame"].tolist() == [0, 1]
    assert table["label"].tolist() == [1, 1]
    assert table["length_px_moments"].notna().all()
    assert "length_px_morphometrics" in table.columns
    assert "method_morphometrics" in table.columns


def test_stack_regionprops_table_rejects_unknown_moments_backend() -> None:
    stack = np.zeros((1, 1, 5, 5), dtype=np.uint8)

    try:
        stack_regionprops_table(
            stack,
            index_names=("sample", "frame"),
            moments_backend="not-a-backend",  # type: ignore[arg-type]
        )
    except ValueError as error:
        assert "moments_backend" in str(error)
    else:
        raise AssertionError("Expected stack_regionprops_table to reject moments_backend.")


def test_stack_regionprops_table_rejects_execution_keyword() -> None:
    stack = np.zeros((1, 1, 8, 8), dtype=np.uint8)
    stack[0, 0, 2:6, 2:6] = 1

    try:
        stack_regionprops_table(
            stack,
            index_names=("sample", "frame"),
            properties=("label", "area"),
            execution="loop",  # type: ignore[call-arg]
        )
    except TypeError as error:
        assert "execution" in str(error)
    else:
        raise AssertionError("Expected stack_regionprops_table to reject execution.")


def test_stack_regionprops_table_vectorizes_moments_when_morphometrics_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = np.zeros((1, 2, 20, 20), dtype=np.uint8)
    stack[0, 0, 5:15, 7:13] = 1
    stack[0, 1, 4:16, 7:13] = 1
    calls: list[tuple[str, ...]] = []
    original = core._stack_regionprops_table_vectorized

    def spy_vectorized(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs["properties"])
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_stack_regionprops_table_vectorized", spy_vectorized)

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "moments_axis", "morphometrics"),
        morphometrics_n_jobs=0,
    )

    assert calls == [("label", "moments_axis")]
    assert "length_px_moments" in table.columns
    assert "length_px_morphometrics" in table.columns
    assert table["label"].tolist() == [1, 1]


def test_stack_regionprops_table_parallel_morphometrics_returns_rows() -> None:
    stack = np.zeros((1, 2, 20, 20), dtype=np.uint8)
    stack[0, 0, 5:15, 7:13] = 1
    stack[0, 1, 4:16, 7:13] = 1

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "morphometrics"),
        morphometrics_n_jobs=2,
    )

    assert table["frame"].tolist() == [0, 1]
    assert table["label"].tolist() == [1, 1]
    assert "method_morphometrics" in table.columns


def test_stack_regionprops_table_vectorized_accepts_dask_arrays() -> None:
    stack_np = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    stack_np[0, 0, 1:3, 1:3] = 1
    stack_np[0, 1, 1:4, 1:3] = 1
    stack = da.from_array(stack_np, chunks=(1, 1, 5, 5))

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
    )

    assert table["frame"].tolist() == [0, 1]
    assert table["area_px"].tolist() == [4, 6]
    assert table["length_px_moments"].notna().all()


def test_stack_regionprops_table_numba_accepts_dask_arrays() -> None:
    stack_np = np.zeros((2, 2, 5, 5), dtype=np.uint8)
    stack_np[0, 0, 1:3, 1:3] = 1
    stack_np[0, 1, 1:4, 1:3] = 1
    stack_np[1, 0, 2:5, 2:4] = 1
    stack = da.from_array(stack_np, chunks=(1, 1, 5, 5))

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="numba",
        moments_chunk_size=1,
    )

    assert table[["sample", "frame"]].to_dict("records") == [
        {"sample": 0, "frame": 0},
        {"sample": 0, "frame": 1},
        {"sample": 1, "frame": 0},
    ]
    assert table["area_px"].tolist() == [4, 6, 6]
    assert table["length_px_moments"].notna().all()


def test_stack_regionprops_table_measures_multichannel_intensity() -> None:
    stack = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    stack[0, 0, 0, 1:] = 1
    stack[0, 1, 1, 1:] = 2
    intensity = np.zeros((1, 2, 2, 3, 3), dtype=float)
    intensity[0, 0, 0] = np.arange(9, dtype=float).reshape(3, 3)
    intensity[0, 0, 1] = 10 + np.arange(9, dtype=float).reshape(3, 3)
    intensity[0, 1, 0] = 100 + np.arange(9, dtype=float).reshape(3, 3)
    intensity[0, 1, 1] = 200 + np.arange(9, dtype=float).reshape(3, 3)

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "intensity"),
        intensity_stack=intensity,
        intensity_channels={"phase": 0, "mcherry": 1},
        moments_backend="numba",
        morphometrics_n_jobs=0,
    )

    assert table[["sample", "frame", "label", "area_px"]].to_dict("records") == [
        {"sample": 0, "frame": 0, "label": 1, "area_px": 2},
        {"sample": 0, "frame": 1, "label": 2, "area_px": 2},
    ]
    assert table["phase_intensity_mean"].tolist() == [1.5, 104.5]
    assert table["mcherry_intensity_mean"].tolist() == [11.5, 204.5]
    assert "phase_intensity_q05" in table.columns
    assert "mcherry_intensity_q95" in table.columns


def test_stack_regionprops_table_accepts_dask_intensity_stack() -> None:
    stack_np = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    stack_np[0, 0, 0, 1:] = 1
    stack_np[0, 1, 1, 1:] = 1
    intensity_np = np.zeros((1, 2, 3, 3), dtype=float)
    intensity_np[0, 0] = np.arange(9, dtype=float).reshape(3, 3)
    intensity_np[0, 1] = 10 + np.arange(9, dtype=float).reshape(3, 3)

    table = stack_regionprops_table(
        da.from_array(stack_np, chunks=(1, 1, 3, 3)),
        index_names=("sample", "frame"),
        properties=("label", "intensity"),
        intensity_stack=da.from_array(intensity_np, chunks=(1, 1, 3, 3)),
        intensity_channels={"phase": 0},
        morphometrics_n_jobs=0,
    )

    assert table[["sample", "frame", "label"]].to_dict("records") == [
        {"sample": 0, "frame": 0, "label": 1},
        {"sample": 0, "frame": 1, "label": 1},
    ]
    assert table["phase_intensity_mean"].tolist() == [1.5, 14.5]


def test_stack_regionprops_table_rejects_intensity_shape_mismatch() -> None:
    stack = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    intensity = np.zeros((1, 3, 3), dtype=float)

    try:
        stack_regionprops_table(
            stack,
            index_names=("sample", "frame"),
            properties=("label", "intensity"),
            intensity_stack=intensity,
            intensity_channels={"phase": 0},
        )
    except ValueError as error:
        assert "intensity_stack" in str(error)
    else:
        raise AssertionError("Expected intensity_stack shape mismatch to be rejected.")
