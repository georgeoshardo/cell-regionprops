# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cell-regionprops",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "zarr",
# ]
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import zarr

    from cell_regionprops import (
        binary_regionprops_table,
        regionprops_table,
        stack_regionprops_table,
    )

    return (
        Path,
        binary_regionprops_table,
        mo,
        np,
        plt,
        regionprops_table,
        stack_regionprops_table,
        zarr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # `cell-regionprops` API demo

    This notebook uses a small real-mask zarr copied from the parent scientific-data repository. The package itself only measures masks; it does not know about the experiment, hypotheses, mothers, or divisions.
    """)
    return


@app.cell
def _(Path, np, zarr):
    demo_zarr = Path(__file__).resolve().parents[1] / "examples" / "data" / "demo_label_masks.zarr"
    demo_masks = np.asarray(zarr.open_array(demo_zarr, mode="r"))
    label_image = demo_masks[0, 0]
    return demo_masks, demo_zarr, label_image


@app.cell
def _(demo_masks, demo_zarr, mo):
    mo.md(f"""
    Demo zarr: `{demo_zarr}`

    Shape: `{demo_masks.shape}` as `(sample, frame, y, x)`
    """)
    return


@app.cell
def _(label_image, plt):
    mask_fig, mask_ax = plt.subplots(figsize=(3.0, 5.0))
    mask_ax.imshow(label_image, cmap="tab20", interpolation="nearest")
    mask_ax.set_title("Example labeled mask")
    mask_ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(label_image, regionprops_table):
    single_frame_table = regionprops_table(
        label_image,
        properties=("label", "area", "centroid", "moments_axis"),
    )
    single_frame_table
    return


@app.cell
def _(label_image, regionprops_table):
    morphometrics_table = regionprops_table(
        label_image,
        properties=("label", "morphometrics", "area", "centroid", "moments_axis"),
    )
    morphometrics_table
    return (morphometrics_table,)


@app.cell
def _(demo_masks, label_image, morphometrics_table, plt):
    _valid_morphometrics = morphometrics_table[
        morphometrics_table["method_morphometrics"] == "contour_voronoi_rib_intersections"
    ]
    _morph_row = _valid_morphometrics.iloc[0]
    _morph_label = int(_morph_row["label"])
    _morph_mesh = _morph_row["mesh_px_morphometrics"]
    _morph_centerline = _morph_row["centerline_xy_morphometrics"]

    morphometrics_mesh_fig, morphometrics_mesh_axes = plt.subplots(1, 2, figsize=(6.0, 5.0))
    _mesh_ax, _frame_250_ax = morphometrics_mesh_axes

    _mesh_ax.imshow(label_image, cmap="tab20", interpolation="nearest")
    for _rib in _morph_mesh:
        _mesh_ax.plot(
            [_rib[0], _rib[2]],
            [_rib[1], _rib[3]],
            color="deepskyblue",
            linewidth=0.8,
            alpha=0.8,
        )
    _mesh_ax.plot(
        _morph_centerline[:, 0],
        _morph_centerline[:, 1],
        color="red",
        linewidth=1.2,
    )
    _mesh_ax.set_title(f"Frame 0, label {_morph_label} mesh")
    _mesh_ax.set_axis_off()

    _frame_250_ax.imshow(demo_masks[0, 250], cmap="tab20", interpolation="nearest")
    _frame_250_ax.set_title("Frame 250")
    _frame_250_ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    morphometrics_mesh_fig
    return


@app.cell
def _(binary_regionprops_table, label_image):
    binary_table = binary_regionprops_table(
        label_image > 0,
        properties=("label", "area", "moments_axis"),
    )
    binary_table
    return


@app.cell
def _(demo_masks, np, stack_regionprops_table):
    _label_one_stack = np.where(demo_masks[0:1, :] == 1, 1, 0)
    stack_table = stack_regionprops_table(
        _label_one_stack,
        index_names=("sample", "frame"),
        properties=("label", "moments_axis", "morphometrics"),
    )
    stack_table.head()
    return (stack_table,)


@app.cell
def _(plt, stack_table):
    length_trace = stack_table.sort_values(["sample", "frame"])

    length_fig, length_ax = plt.subplots(figsize=(5.0, 3.0))
    length_ax.plot(
        length_trace["frame"],
        length_trace["length_moments"],
        marker="o",
        markersize=2,
        linewidth=1.0,
        label="moments",
    )
    length_ax.plot(
        length_trace["frame"],
        length_trace["length_morphometrics"],
        marker="o",
        markersize=2,
        linewidth=1.0,
        label="morphometrics",
    )
    length_ax.set_xlabel("Frame")
    length_ax.set_ylabel("Length (px)")
    length_ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    length_fig
    return


if __name__ == "__main__":
    app.run()
