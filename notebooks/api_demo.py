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

__generated_with = "0.23.0"
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
        pd,
        plt,
        regionprops_table,
        stack_regionprops_table,
        zarr,
    )


@app.cell
def _(mo):
    mo.Html(
        "<h1><code>cell-regionprops</code> API demo</h1>"
        "<p>This notebook uses a small real-mask zarr copied from the parent "
        "scientific-data repository. The package itself only measures masks; it "
        "does not know about the experiment, hypotheses, mothers, or divisions.</p>"
    )
    return


@app.cell
def _(Path, np, zarr):
    demo_zarr = Path(__file__).resolve().parents[1] / "examples" / "data" / "demo_label_masks.zarr"
    demo_masks = np.asarray(zarr.open_array(demo_zarr, mode="r"))
    label_image = demo_masks[0, 0]
    return demo_masks, demo_zarr, label_image


@app.cell
def _(demo_masks, demo_zarr, mo):
    mo.Html(
        f"<p>Demo zarr: <code>{demo_zarr}</code></p>"
        f"<p>Shape: <code>{demo_masks.shape}</code> as "
        "<code>(sample, frame, y, x)</code></p>"
    )
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
    single_frame_table.head()
    return (single_frame_table,)


@app.cell
def _(binary_regionprops_table, label_image):
    binary_table = binary_regionprops_table(
        label_image > 0,
        properties=("label", "area", "moments_axis"),
    )
    binary_table
    return (binary_table,)


@app.cell
def _(demo_masks, stack_regionprops_table):
    stack_table = stack_regionprops_table(
        demo_masks[:2, :5],
        index_names=("sample", "frame"),
        properties=("label", "area"),
    )
    stack_table.head()
    return (stack_table,)


@app.cell
def _(plt, stack_table):
    area_trace = (
        stack_table.groupby(["sample", "frame"], as_index=False)["area_px"]
        .sum()
        .sort_values(["sample", "frame"])
    )

    area_fig, area_ax = plt.subplots(figsize=(5.0, 3.0))
    for sample, group in area_trace.groupby("sample"):
        area_ax.plot(group["frame"], group["area_px"], marker="o", label=f"sample {sample}")
    area_ax.set_xlabel("Frame")
    area_ax.set_ylabel("Total labeled area (px)")
    area_ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    area_fig
    return (area_trace,)


if __name__ == "__main__":
    app.run()
