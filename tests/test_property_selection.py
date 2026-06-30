import numpy as np

from cell_regionprops import regionprops_table


def test_only_requested_columns_are_returned() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    table = regionprops_table(label_image, properties=("label",))

    assert list(table.columns) == ["label"]
    assert table["label"].tolist() == [1]


def test_missing_label_respects_requested_properties() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    table = regionprops_table(label_image, labels=[1, 3], properties=("label",))

    assert list(table.columns) == ["label"]
    assert table["label"].tolist() == [1, 3]


def test_extra_property_adds_named_column() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    def topmost_row(mask: np.ndarray) -> int:
        rows = np.flatnonzero(mask.any(axis=1))
        return int(rows[0]) if rows.size else -1

    table = regionprops_table(
        label_image,
        properties=("label",),
        extra_properties={"topmost_row": topmost_row},
    )

    assert table["label"].tolist() == [1]
    assert table["topmost_row"].tolist() == [0]

