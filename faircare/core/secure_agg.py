# faircare/core/secure_agg.py
import numpy as np

def mask_vector(v, rng):
    """One-time pad style mask for demo secure aggregation."""
    mask = rng.integers(low=-2**31, high=2**31, size=v.shape, dtype=np.int64)
    return v.astype(np.int64) + mask, mask

def unmask_sum(masked_vs, masks):
    """Sum masked vectors and subtract masks (demo only; not a real protocol)."""
    total = np.sum(masked_vs, axis=0).astype(np.int64)
    total_mask = np.sum(masks, axis=0).astype(np.int64)
    return (total - total_mask).astype(np.int64)
