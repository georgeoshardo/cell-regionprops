import numpy as np
from skimage.measure import regionprops_table as skimage_regionprops_table

from cell_regionprops.skimage_extra import make_extra_properties


def test_make_extra_properties_works_with_skimage_regionprops_table() -> None:
    label_image = np.zeros((40, 40), dtype=np.uint8)
    label_image[18:22, 10:30] = 1

    extra_properties = make_extra_properties(
        ("moments_axis_length", "moments_axis_width"),
        pixel_size=0.5,
    )
    table = skimage_regionprops_table(
        label_image,
        properties=("label",),
        extra_properties=extra_properties,
    )

    assert "moments_axis_length" in table
    assert "moments_axis_width" in table
    assert table["moments_axis_length"][0] > 0
    assert table["moments_axis_width"][0] > 0


def test_make_extra_properties_can_measure_morphometrics_columns() -> None:
    label_image = np.zeros((40, 40), dtype=np.uint8)
    label_image[18:22, 10:30] = 1

    extra_properties = make_extra_properties(
        ("morphometrics_length", "morphometrics_width"),
        pixel_size=0.5,
    )
    table = skimage_regionprops_table(
        label_image,
        properties=("label",),
        extra_properties=extra_properties,
    )

    assert table["morphometrics_length"][0] > 0
    assert table["morphometrics_width"][0] > 0

