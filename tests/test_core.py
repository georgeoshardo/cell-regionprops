import numpy as np
import pandas as pd

from cell_regionprops import binary_regionprops_table, regionprops_table


def test_regionprops_table_returns_one_row_per_label() -> None:
    label_image = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.uint8,
    )

    table = regionprops_table(label_image, pixel_size=0.5)

    assert table["label"].tolist() == [1, 2]
    assert table["area_px"].tolist() == [3, 3]
    assert table["area"].tolist() == [0.75, 0.75]
    assert table["method"].tolist() == ["moments", "moments"]
    assert table["length_px"].notna().all()
    assert table["width_px"].notna().all()


def test_regionprops_table_preserves_requested_missing_labels() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    table = regionprops_table(label_image, labels=[1, 3])

    assert table["label"].tolist() == [1, 3]

    missing = table.loc[table["label"] == 3].iloc[0]
    assert missing["area_px"] == 0
    assert missing["area"] == 0
    assert pd.isna(missing["centroid_y"])
    assert pd.isna(missing["centroid_x"])
    assert pd.isna(missing["length_px"])
    assert pd.isna(missing["width_px"])
    assert missing["method"] == "missing"


def test_binary_regionprops_table_uses_foreground_as_label_one() -> None:
    binary_mask = np.array(
        [
            [False, True, True],
            [False, False, True],
        ]
    )

    table = binary_regionprops_table(binary_mask, pixel_size=2.0)

    assert table["label"].tolist() == [1]
    assert table["area_px"].tolist() == [3]
    assert table["area"].tolist() == [12.0]

