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
    assert "method" not in table.columns
    assert "length_px" not in table.columns
    assert "width_px" not in table.columns
    assert table["length_px_moments"].notna().all()
    assert table["width_px_moments"].notna().all()


def test_regionprops_table_preserves_requested_missing_labels() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    table = regionprops_table(label_image, labels=[1, 3])

    assert table["label"].tolist() == [1, 3]

    missing = table.loc[table["label"] == 3].iloc[0]
    assert missing["area_px"] == 0
    assert missing["area"] == 0
    assert pd.isna(missing["centroid_y"])
    assert pd.isna(missing["centroid_x"])
    assert pd.isna(missing["length_px_moments"])
    assert pd.isna(missing["width_px_moments"])
    assert pd.isna(missing["length_moments"])
    assert pd.isna(missing["width_moments"])


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


def test_regionprops_table_measures_single_channel_intensity() -> None:
    label_image = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    intensity_image = np.array(
        [
            [0, 10, 20],
            [0, 0, 30],
        ],
        dtype=float,
    )

    table = regionprops_table(
        label_image,
        properties=("label", "intensity"),
        intensity_image=intensity_image,
        intensity_channels={"phase": 0},
    )

    row = table.iloc[0]
    assert row["phase_intensity_mean"] == 20.0
    assert row["phase_intensity_median"] == 20.0
    assert row["phase_intensity_min"] == 10.0
    assert row["phase_intensity_max"] == 30.0
    assert row["phase_intensity_q05"] == 11.0
    assert row["phase_intensity_q95"] == 29.0


def test_regionprops_table_measures_multichannel_intensity() -> None:
    label_image = np.array(
        [
            [1, 1],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    intensity_image = np.array(
        [
            [[1, 2], [3, 4]],
            [[10, 20], [30, 40]],
        ],
        dtype=float,
    )

    table = regionprops_table(
        label_image,
        properties=("label", "intensity"),
        intensity_image=intensity_image,
        intensity_channels={"phase": 0, "mcherry": 1},
    )

    row = table.iloc[0]
    assert row["phase_intensity_mean"] == 7.0 / 3.0
    assert row["phase_intensity_min"] == 1.0
    assert row["phase_intensity_max"] == 4.0
    assert row["mcherry_intensity_mean"] == 70.0 / 3.0
    assert row["mcherry_intensity_min"] == 10.0
    assert row["mcherry_intensity_max"] == 40.0


def test_regionprops_table_missing_label_has_nan_intensity() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    intensity_image = np.array([[0, 5], [0, 7]], dtype=float)

    table = regionprops_table(
        label_image,
        labels=[1, 3],
        properties=("label", "area", "intensity"),
        intensity_image=intensity_image,
        intensity_channels={"phase": 0},
    )

    missing = table.loc[table["label"] == 3].iloc[0]
    assert missing["area_px"] == 0
    assert pd.isna(missing["phase_intensity_mean"])
    assert pd.isna(missing["phase_intensity_q05"])


def test_regionprops_table_rejects_intensity_without_image() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    try:
        regionprops_table(
            label_image,
            properties=("label", "intensity"),
            intensity_channels={"phase": 0},
        )
    except ValueError as error:
        assert "intensity_image" in str(error)
    else:
        raise AssertionError("Expected missing intensity_image to be rejected.")


def test_regionprops_table_rejects_bad_intensity_channel() -> None:
    label_image = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    intensity_image = np.zeros((1, 2, 2), dtype=float)

    try:
        regionprops_table(
            label_image,
            properties=("label", "intensity"),
            intensity_image=intensity_image,
            intensity_channels={"phase": 2},
        )
    except ValueError as error:
        assert "out-of-bounds" in str(error)
    else:
        raise AssertionError("Expected out-of-bounds intensity channel to be rejected.")
