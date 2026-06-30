# cell-regionprops

`regionprops_table`-style measurements for labeled masks of cells.

## Tiny Example

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from cell_regionprops import plot_meshes_over_mask, stack_regionprops_table

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

# Plot the contour-rib mesh for one frame.
colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
colors[0] = [0, 0, 0, 1]
mask_cmap = plt.matplotlib.colors.ListedColormap(colors)

frame_table = table.query("sample == 0 and frame == 0")
fig, ax = plt.subplots(figsize=(3.0, 5.0))
plot_meshes_over_mask(masks[0, 0], frame_table, ax=ax, mask_cmap=mask_cmap)
plt.show()
```

`table.head()`:

| sample | frame | label | area_px | centroid_y | centroid_x | length_px_moments | width_px_moments | length_px_morphometrics | width_px_morphometrics | method_morphometrics |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 | 0 | 1 | 101 | 25.36 | 18.92 | 14.82 | 8.66 | 14.35 | 8.24 | contour_voronoi_rib_intersections |
| 0 | 0 | 2 | 146 | 42.03 | 22.20 | 26.08 | 7.38 | 23.34 | 6.36 | contour_voronoi_rib_intersections |
| 0 | 0 | 3 | 116 | 41.85 | 15.35 | 18.39 | 8.04 | 18.06 | 7.81 | contour_voronoi_rib_intersections |
| 0 | 0 | 4 | 98 | 57.49 | 18.94 | 15.98 | 7.87 | 15.13 | 7.20 | contour_voronoi_rib_intersections |
| 0 | 0 | 5 | 106 | 66.46 | 18.42 | 17.31 | 7.88 | 16.63 | 7.28 | contour_voronoi_rib_intersections |

![Mesh overlay example](docs/assets/readme_mesh_overlay.png)
