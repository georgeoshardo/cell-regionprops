import numpy as np
import dask.array as da

from cell_regionprops import stack_regionprops_table


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


def test_stack_regionprops_table_can_require_vectorized_moments_path() -> None:
    stack = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    stack[0, 0, 1:3, 1:3] = 1
    stack[0, 1, 1:4, 1:3] = 1

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        execution="vectorized",
    )

    assert table["frame"].tolist() == [0, 1]
    assert table["label"].tolist() == [1, 1]
    assert table["area_px"].tolist() == [4, 6]
    assert table["length_px_moments"].notna().all()
    assert table["width_px_moments"].notna().all()


def test_stack_regionprops_table_vectorized_matches_loop_for_moments() -> None:
    stack = np.zeros((2, 3, 6, 6), dtype=np.uint8)
    stack[0, 0, 1:3, 1:4] = 1
    stack[0, 1, 2:5, 2:4] = 2
    stack[1, 2, 1:5, 1:3] = 1

    vectorized = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        execution="vectorized",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)
    loop = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        execution="loop",
    ).sort_values(["sample", "frame", "label"]).reset_index(drop=True)

    assert vectorized.columns.tolist() == loop.columns.tolist()
    np.testing.assert_allclose(
        vectorized.drop(columns=["sample", "frame", "label"]).to_numpy(dtype=float),
        loop.drop(columns=["sample", "frame", "label"]).to_numpy(dtype=float),
    )


def test_stack_regionprops_table_rejects_unsupported_vectorized_properties() -> None:
    stack = np.zeros((1, 1, 8, 8), dtype=np.uint8)
    stack[0, 0, 2:6, 2:6] = 1

    try:
        stack_regionprops_table(
            stack,
            index_names=("sample", "frame"),
            properties=("label", "morphometrics"),
            execution="vectorized",
        )
    except ValueError as error:
        assert "vectorized" in str(error)
    else:
        raise AssertionError("Expected unsupported vectorized properties to fail.")


def test_stack_regionprops_table_vectorized_accepts_dask_arrays() -> None:
    stack_np = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    stack_np[0, 0, 1:3, 1:3] = 1
    stack_np[0, 1, 1:4, 1:3] = 1
    stack = da.from_array(stack_np, chunks=(1, 1, 5, 5))

    table = stack_regionprops_table(
        stack,
        index_names=("sample", "frame"),
        properties=("label", "area", "centroid", "moments_axis"),
        execution="vectorized",
    )

    assert table["frame"].tolist() == [0, 1]
    assert table["area_px"].tolist() == [4, 6]
    assert table["length_px_moments"].notna().all()
