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
    mask_fig
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
        properties=("label", "morphometrics"),
    )
    morphometrics_table
    return (morphometrics_table,)


@app.cell
def _(label_image, morphometrics_table, np, plt):
    _valid_morphometrics = morphometrics_table[
        morphometrics_table["method_morphometrics"] == "contour_voronoi_rib_intersections"
    ]
    _morph_row = _valid_morphometrics.iloc[0]
    _morph_label = int(_morph_row["label"])
    _morph_mask = np.where(label_image == _morph_label, _morph_label, 0)
    _morph_mesh = _morph_row["mesh_px_morphometrics"]
    _morph_centerline = _morph_row["centerline_xy_morphometrics"]

    morphometrics_mesh_fig, morphometrics_mesh_ax = plt.subplots(figsize=(3.0, 5.0))
    morphometrics_mesh_ax.imshow(_morph_mask, cmap="tab20", interpolation="nearest")
    for _rib in _morph_mesh:
        morphometrics_mesh_ax.plot(
            [_rib[0], _rib[2]],
            [_rib[1], _rib[3]],
            color="deepskyblue",
            linewidth=0.8,
            alpha=0.8,
        )
    morphometrics_mesh_ax.plot(
        _morph_centerline[:, 0],
        _morph_centerline[:, 1],
        color="red",
        linewidth=1.2,
    )
    morphometrics_mesh_ax.set_title(f"Label {_morph_label}: Morphometrics mesh")
    morphometrics_mesh_ax.set_axis_off()
    plt.tight_layout()
    plt.show()
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
def _(demo_masks, stack_regionprops_table):
    stack_table = stack_regionprops_table(
        demo_masks[0:1, :],
        index_names=("sample", "frame"),
        properties=("label", "area"),
    )
    stack_table.head()
    return (stack_table,)


@app.cell
def _(plt, stack_table):
    area_trace = (
        stack_table.groupby(["sample", "frame", "label"], as_index=False)["area_px"]
        .sum()
        .sort_values(["sample", "frame"])
    )

    mother_cell_area = area_trace.query("label == 1")

    area_fig, area_ax = plt.subplots(figsize=(5.0, 3.0))
    area_ax.plot(mother_cell_area["frame"], mother_cell_area["area_px"], marker="o")
    area_ax.set_xlabel("Frame")
    area_ax.set_ylabel("Total labeled area (px)")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
