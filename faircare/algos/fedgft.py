from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from .aggregator import BaseAggregator

class FedGFTAggregator(BaseAggregator):
    """
    Use a global fairness penalty proxy via 'factor' (server passes in factors that
    summarize global EO/SP gaps from last round). We emphasize clients that contain
    more of underperforming groups (encoded in factor<1).
    """
    def __init__(self, sens_present: bool, kappa: float = 0.4):
        super().__init__(sens_present)
        self.kappa = kappa

    def compute_weights(self, local_meta: List[Dict[str, Any]]) -> List[float]:
        n = len(local_meta)
        f = np.array([m["factor"] for m in local_meta])
        w = np.exp(self.kappa * (f.max() - f))
        w = w / w.sum()
        return w.tolist()

def make_aggregator(sens_present: bool) -> BaseAggregator:
    return FedGFTAggregator(sens_present)
