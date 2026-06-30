import numpy as np

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

