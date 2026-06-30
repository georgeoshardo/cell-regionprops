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

![Mesh overlay example](docs/assets/readme_mesh_overlay.png)
