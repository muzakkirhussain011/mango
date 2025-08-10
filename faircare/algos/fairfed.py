from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from .aggregator import BaseAggregator

class FairFedAggregator(BaseAggregator):
    """
    Heuristic: increase weights for clients that appear to represent underperforming groups
    by relying on server-provided local 'factor' (lower factor -> underperforming).
    """
    def __init__(self, sens_present: bool, gamma: float = 0.3):
        super().__init__(sens_present)
        self.gamma = gamma

    def compute_weights(self, local_meta: List[Dict[str, Any]]) -> List[float]:
        n = len(local_meta)
        f = np.array([m["factor"] for m in local_meta])
        # higher weight to smaller factors
        w = (1.0 + self.gamma * (f.max() - f))  # linear boost
        w = w / w.sum()
        return w.tolist()

def make_aggregator(sens_present: bool) -> BaseAggregator:
    return FairFedAggregator(sens_present)
