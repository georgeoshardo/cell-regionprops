"""Create a small real-mask zarr fixture for the API demo notebook."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PARENT_REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ZARR = PARENT_REPO_ROOT / "20260307_SB7_exit_snake_V4_1.segmentation_masks_multi_epoch_uint8.zarr"
OUTPUT_ZARR = PACKAGE_ROOT / "examples" / "data" / "demo_label_masks.zarr"

SOURCE_HYPOTHESIS_INDEX = 16
N_SAMPLES = 8
RANDOM_SEED = 20260630


def main() -> None:
    """Copy a small `(sample, frame, y, x)` mask stack into the package."""
    source = zarr.open(SOURCE_ZARR, mode="r")["data"]
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = np.sort(rng.choice(source.shape[1], size=N_SAMPLES, replace=False))
    demo_masks = np.asarray(
        source[
            SOURCE_HYPOTHESIS_INDEX,
            sample_indices,
            :,
            :,
            :,
        ],
        dtype=np.uint8,
    )

    if OUTPUT_ZARR.exists():
        shutil.rmtree(OUTPUT_ZARR)
    OUTPUT_ZARR.parent.mkdir(parents=True, exist_ok=True)

    output = zarr.open_array(
        OUTPUT_ZARR,
        mode="w",
        shape=demo_masks.shape,
        dtype=demo_masks.dtype,
        chunks=(1, demo_masks.shape[1], demo_masks.shape[2], demo_masks.shape[3]),
    )
    output[:] = demo_masks
    output.attrs.clear()

    print(f"Wrote {OUTPUT_ZARR}")
    print(f"shape={output.shape}, dtype={output.dtype}")


if __name__ == "__main__":
    main()
