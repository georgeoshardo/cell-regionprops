import numpy as np

from cell_regionprops import regionprops_table
from cell_regionprops.morphometrics_mesh import measure_cell_mesh


def test_measure_cell_mesh_uses_contour_voronoi_for_simple_cell() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 10:30] = True

    result = measure_cell_mesh(mask, pixel_size_um=0.1)

    assert result["method"] == "contour_voronoi_rib_intersections"
    assert result["length_px"] > 0
    assert result["width_px"] > 0
    assert result["length_um"] > 0
    assert result["width_um"] > 0
    assert result["mesh_px"].shape[1] == 4


def test_measure_cell_mesh_returns_failed_measurement_for_tiny_masks() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[9:11, 9:11] = True

    result = measure_cell_mesh(mask)

    assert result["method"] == "contour_voronoi_failed"
    assert np.isnan(result["length_px"])
    assert result["mesh_px"].shape == (0, 4)


def test_regionprops_table_can_measure_morphometrics_property() -> None:
    label_image = np.zeros((40, 40), dtype=np.uint8)
    label_image[18:22, 10:30] = 1

    table = regionprops_table(label_image, properties=("label", "morphometrics"))

    assert table["label"].tolist() == [1]
    assert table["method_morphometrics"].tolist() == ["contour_voronoi_rib_intersections"]
    assert table["length_px_morphometrics"].iloc[0] > 0
    assert table["width_px_morphometrics"].iloc[0] > 0
    assert table["volume_morphometrics"].iloc[0] > 0
    assert table["surface_area_morphometrics"].iloc[0] > 0
    assert table["surface_area_to_volume_ratio_morphometrics"].iloc[0] > 0

