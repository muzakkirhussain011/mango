# faircare/core/secure_agg.py
import numpy as np

def mask_vector(v, rng):
    """One-time pad style mask for demo secure aggregation."""
    mask = rng.integers(low=-2**31, high=2**31, size=v.shape, dtype=np.int64)
    return v.astype(np.int64) + mask, mask

def unmask_sum(masked_vs, masks):
    """Sum masked vectors and subtract masks (demo only; not afrom __future__ import annotations
from typing import List
import torch

class SecureAggregator:
    """
    Simple additive-masking secure aggregation:
    - Each client i submits (update_i + r_i - r_{i-1}), masks cancel in sum.
    - Here we simulate by directly summing updates (since masks cancel),
      but interface mirrors a real secure-agg pipeline for drop-in replacement.
    """
    def __init__(self): pass

    def aggregate(self, updates: List[torch.Tensor]) -> torch.Tensor:
        if len(updates) == 0:
            raise ValueError("No updates to aggregate")
        stacked = torch.stack(updates, dim=0)
        return torch.sum(stacked, dim=0) / len(updates)
 real protocol)."""
    total = np.sum(masked_vs, axis=0).astype(np.int64)
    total_mask = np.sum(masks, axis=0).astype(np.int64)
    return (total - total_mask).astype(np.int64)
