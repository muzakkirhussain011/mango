from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from .aggregator import BaseAggregator

class FairCareAggregator(BaseAggregator):
    """
    Unified multi-level fairness:
    - 'factor' encodes server-side importance (client accuracy gap + group gap attribution).
    - We combine client-level dispersion and group-level deficit into a single scaling.
    """
    def __init__(self, sens_present: bool, alpha_c: float = 0.5, alpha_g: float = 0.5):
        super().__init__(sens_present)
        self.alpha_c = alpha_c
        self.alpha_g = alpha_g

    def compute_weights(self, local_meta: List[Dict[str, Any]]) -> List[float]:
        n = len(local_meta)
        f = np.array([m["factor"] for m in local_meta])  # lower -> more underperforming
        # convert into importance: higher for lower factor
        imp = (self.alpha_c + self.alpha_g) * (f.max() - f + 1e-6)
        imp = imp + 1.0  # keep bounded away from zero
        imp = imp / imp.sum()
        return imp.tolist()

    def client_weights_signal(self) -> List[float]:
        # return a neutral signal; server fills with concrete values per round
        return [1.0 for _ in range(10)]

def make_aggregator(sens_present: bool) -> BaseAggregator:
    return FairCareAggregator(sens_present)
