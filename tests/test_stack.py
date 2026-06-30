import numpy as np
import dask.array as da
import pandas as pd
import pytest

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


def test_stack_regionprops_table_bincount_matches_label_loop_for_moments() -> None:
    stack = np.zeros((2, 3, 7, 8), dtype=np.int32)
    stack[0, 0, 1:3, 1:5] = 1
    stack[0, 0, 4:6, 2:7] = 1000
    stack[0, 1, 2:6, 3:5] = 1
    stack[1, 0, 1:6, 1:3] = 1000
    stack[1, 2, 3:6, 4:7] = 1

    label_loop = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="label_loop",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)
    bincount = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        moments_backend="bincount",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)

    assert bincount.columns.tolist() == label_loop.columns.tolist()
    assert bincount[["sample", "frame", "label", "area_px"]].equals(
        label_loop[["sample", "frame", "label", "area_px"]]
    )
    np.testing.assert_allclose(
        bincount.drop(columns=["sample", "frame", "label", "area_px"]).to_numpy(dtype=float),
        label_loop.drop(columns=["sample", "frame", "label", "area_px"]).to_numpy(dtype=float),
    )


def test_stack_regionprops_table_bincount_merges_with_morphometrics() -> None:
    stack = np.zeros((1, 2, 20, 20), dtype=np.uint8)
    stack[0, 0, 5:15, 7:13] = 1
    stack[0, 1, 4:16, 7:13] = 1

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "moments_axis", "morphometrics"),
        moments_backend="bincount",
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

    def spy_vectorized(*args, **kwargs):
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
