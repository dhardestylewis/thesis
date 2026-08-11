"""Repack inference_cache.npy as a headerless float32 binary for HTTP Range serving.

Layout is [height_tier][parcel][delay, attrition, withdrawal], C-order, so the
slice for one height tier is contiguous: offset = tier * parcels * 3 * 4 bytes.
"""
import numpy as np
from pathlib import Path

DATA = Path(r"C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases")
grid = np.load(DATA / "inference_cache.npy", mmap_mode='r')
n_heights, n_parcels, n_metrics = grid.shape
out = DATA / "austin_friction_grid.f32"

slice_bytes = n_parcels * n_metrics * 4
print(f"grid {grid.shape}  slice {slice_bytes:,} bytes  total {slice_bytes * n_heights:,}", flush=True)

with open(out, 'wb') as f:
    for i in range(n_heights):
        np.ascontiguousarray(grid[i], dtype=np.float32).tofile(f)
        if (i + 1) % 20 == 0:
            print(f"  wrote tier {i + 1}/{n_heights}", flush=True)

size = out.stat().st_size
assert size == slice_bytes * n_heights, (size, slice_bytes * n_heights)

# Verify a middle tier round-trips exactly
tier = 55
got = np.fromfile(out, dtype=np.float32, count=n_parcels * 3, offset=tier * slice_bytes)
assert np.array_equal(got, np.asarray(grid[tier], dtype=np.float32).ravel())
print(f"OK {out.name}: {size:,} bytes, tier {tier} verified byte-exact", flush=True)
print(f"manifest: heights 5..{4 + n_heights} ft, parcels {n_parcels}, slice {slice_bytes}", flush=True)
