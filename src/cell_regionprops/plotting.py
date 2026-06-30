"""Plotting helpers for measured cell meshes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def _iter_mesh_records(
    meshes: pd.DataFrame | Mapping[str, object] | Iterable[Mapping[str, object]] | ArrayLike,
    *,
    mesh_column: str,
    centerline_column: str,
    method_column: str,
    valid_method: str,
) -> Iterable[tuple[NDArray[np.float64], NDArray[np.float64] | None]]:
    if isinstance(meshes, pd.DataFrame):
        records = meshes.to_dict("records")
    elif isinstance(meshes, Mapping):
        records = [meshes]
    else:
        mesh_array = np.asarray(meshes)
        if mesh_array.ndim == 2 and mesh_array.shape[1] == 4:
            yield mesh_array.astype(float), None
            return
        records = list(meshes)  # type: ignore[arg-type]

    for record in records:
        if method_column in record and record[method_column] != valid_method:
            continue
        mesh = np.asarray(record[mesh_column], dtype=float)
        centerline = (
            np.asarray(record[centerline_column], dtype=float)
            if centerline_column in record
            else None
        )
        yield mesh, centerline


def _mesh_centerline(mesh: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack(
        [
            (mesh[:, 0] + mesh[:, 2]) / 2.0,
            (mesh[:, 1] + mesh[:, 3]) / 2.0,
        ]
    )


def plot_meshes_over_mask(
    label_image: ArrayLike,
    meshes: pd.DataFrame | Mapping[str, object] | Iterable[Mapping[str, object]] | ArrayLike,
    *,
    ax: Any | None = None,
    mask_cmap: Any = "gray",
    mesh_column: str = "mesh_px_morphometrics",
    centerline_column: str = "centerline_xy_morphometrics",
    method_column: str = "method_morphometrics",
    valid_method: str = "contour_voronoi_rib_intersections",
    rib_color: str = "white",
    centerline_color: str = "yellow",
    outline_color: str = "black",
    rib_linewidth: float = 1.0,
    centerline_linewidth: float = 1.4,
    outline_linewidth: float = 2.6,
    alpha: float = 0.95,
) -> Any:
    """Plot one labeled mask with one or more mesh overlays.

    Args:
        label_image: Two-dimensional labeled or binary mask to display.
        meshes: Mesh data to overlay. This can be a dataframe from
            `regionprops_table(..., properties=("morphometrics",))`, one row-like
            mapping, an iterable of row-like mappings, or a single `(n, 4)` mesh
            array.
        ax: Optional matplotlib axes. A new axes is created when omitted.
        mask_cmap: Matplotlib colormap used for the mask image.
        mesh_column: Column/key containing `(n, 4)` mesh rib endpoints.
        centerline_column: Optional column/key containing `(n, 2)` centerline
            coordinates.
        method_column: Column/key containing the morphometrics method string.
        valid_method: Method value to plot when `method_column` is present.
        rib_color: Color for the mesh ribs.
        centerline_color: Color for the mesh centerline.
        outline_color: Color used below ribs and centerlines for contrast.
        rib_linewidth: Width of the visible rib lines.
        centerline_linewidth: Width of the visible centerline.
        outline_linewidth: Width of the contrast outline.
        alpha: Overlay opacity.

    Returns:
        The matplotlib axes containing the mask and mesh overlay.
    """
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(np.asarray(label_image), cmap=mask_cmap, interpolation="nearest")
    ax.set_axis_off()

    for mesh, centerline in _iter_mesh_records(
        meshes,
        mesh_column=mesh_column,
        centerline_column=centerline_column,
        method_column=method_column,
        valid_method=valid_method,
    ):
        if mesh.size == 0:
            continue
        segments = [segment for segment in mesh.reshape((-1, 2, 2))]
        ax.add_collection(
            LineCollection(
                segments,
                colors=outline_color,
                linewidths=outline_linewidth,
                alpha=alpha,
            )
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=rib_color,
                linewidths=rib_linewidth,
                alpha=alpha,
            )
        )

        plotted_centerline = centerline if centerline is not None else _mesh_centerline(mesh)
        if plotted_centerline.size == 0:
            continue
        ax.plot(
            plotted_centerline[:, 0],
            plotted_centerline[:, 1],
            color=outline_color,
            linewidth=outline_linewidth,
            alpha=alpha,
        )
        ax.plot(
            plotted_centerline[:, 0],
            plotted_centerline[:, 1],
            color=centerline_color,
            linewidth=centerline_linewidth,
            alpha=alpha,
        )

    return ax
