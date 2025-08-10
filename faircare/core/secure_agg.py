from __future__ import annotations
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
