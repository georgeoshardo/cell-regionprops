import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from cell_regionprops import plot_meshes_over_mask


def test_plot_meshes_over_mask_draws_mask_and_mesh_without_title() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:5] = 1
    mesh = np.array(
        [
            [3.0, 2.0, 5.0, 2.0],
            [3.0, 3.5, 5.0, 3.5],
            [3.0, 5.0, 5.0, 5.0],
        ]
    )
    centerline = np.array([[4.0, 2.0], [4.0, 3.5], [4.0, 5.0]])
    table = pd.DataFrame(
        {
            "method_morphometrics": ["contour_voronoi_rib_intersections"],
            "mesh_px_morphometrics": [mesh],
            "centerline_xy_morphometrics": [centerline],
        }
    )

    fig, ax = plt.subplots()
    returned_ax = plot_meshes_over_mask(mask, table, ax=ax)

    assert returned_ax is ax
    assert ax.get_title() == ""
    assert len(ax.images) == 1
    assert len(ax.collections) == 2
    assert len(ax.lines) == 2
    assert not ax.axison
    plt.close(fig)


def test_plot_meshes_over_mask_ignores_failed_morphometrics_rows() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:5] = 1
    table = pd.DataFrame(
        {
            "method_morphometrics": ["contour_voronoi_failed"],
            "mesh_px_morphometrics": [np.empty((0, 4))],
            "centerline_xy_morphometrics": [np.empty((0, 2))],
        }
    )

    fig, ax = plt.subplots()
    plot_meshes_over_mask(mask, table, ax=ax)

    assert len(ax.images) == 1
    assert len(ax.collections) == 0
    assert len(ax.lines) == 0
    plt.close(fig)
