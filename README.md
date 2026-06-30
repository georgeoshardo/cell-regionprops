# cell-regionprops

`regionprops_table`-style measurements for labeled masks of cells.

## Tiny Example

```python
from pathlib import Path

import numpy as np
import zarr

from cell_regionprops import stack_regionprops_table

# Load a small real labelled-mask zarr included with the package.
mask_path = Path("examples/data/demo_label_masks.zarr")
masks = np.asarray(zarr.open_array(mask_path, mode="r"))[:1, :30]

# Measure every labelled object in the loaded frames.
table = stack_regionprops_table(
    masks,
    index_names=("sample", "frame"),
    properties=("label", "area", "centroid", "moments_axis", "morphometrics"),
    moments_backend="numba",
    morphometrics_n_jobs=-1,
)

table.head()
```

Array-valued mesh columns are abbreviated by shape.

| sample | frame | label | area_px | area | centroid_y | centroid_x | length_px_moments | width_px_moments | length_moments | width_moments | length_px_morphometrics | width_px_morphometrics | length_morphometrics | width_morphometrics | volume_morphometrics | surface_area_morphometrics | surface_area_to_volume_ratio_morphometrics | method_morphometrics | centerline_xy_morphometrics | width_profile_px_morphometrics | mesh_px_morphometrics |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| 0 | 0 | 1 | 101 | 101.00 | 25.36 | 18.92 | 14.82 | 8.66 | 14.82 | 8.66 | 14.35 | 8.24 | 14.35 | 8.24 | 618.05 | 371.21 | 0.60 | contour_voronoi_rib_intersections | array(8, 2) | array(8,) | array(8, 4) |
| 0 | 0 | 2 | 146 | 146.00 | 42.03 | 22.20 | 26.08 | 7.38 | 26.08 | 7.38 | 23.34 | 6.36 | 23.34 | 6.36 | 673.02 | 465.91 | 0.69 | contour_voronoi_rib_intersections | array(18, 2) | array(18,) | array(18, 4) |
| 0 | 0 | 3 | 116 | 116.00 | 41.85 | 15.35 | 18.39 | 8.04 | 18.39 | 8.04 | 18.06 | 7.81 | 18.06 | 7.81 | 740.13 | 443.03 | 0.60 | contour_voronoi_rib_intersections | array(12, 2) | array(12,) | array(12, 4) |
| 0 | 0 | 4 | 98 | 98.00 | 57.49 | 18.94 | 15.98 | 7.87 | 15.98 | 7.87 | 15.13 | 7.20 | 15.13 | 7.20 | 518.49 | 342.28 | 0.66 | contour_voronoi_rib_intersections | array(6, 2) | array(6,) | array(6, 4) |
| 0 | 0 | 5 | 106 | 106.00 | 66.46 | 18.42 | 17.31 | 7.88 | 17.31 | 7.88 | 16.63 | 7.28 | 16.63 | 7.28 | 591.11 | 380.32 | 0.64 | contour_voronoi_rib_intersections | array(12, 2) | array(12,) | array(12, 4) |

```python
import matplotlib.pyplot as plt

from cell_regionprops import plot_meshes_over_mask

# Plot the contour-rib mesh for one frame.
colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
colors[0] = [0, 0, 0, 1]
mask_cmap = plt.matplotlib.colors.ListedColormap(colors)

frame_table = table.query("sample == 0 and frame == 0")
fig, ax = plt.subplots(figsize=(3.0, 5.0))
plot_meshes_over_mask(masks[0, 0], frame_table, ax=ax, mask_cmap=mask_cmap)
plt.show()
```

![Mesh overlay example](docs/assets/readme_mesh_overlay.png)
