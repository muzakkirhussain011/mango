from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from .aggregator import BaseAggregator

class AFLAggregator(BaseAggregator):
    def __init__(self, sens_present: bool, lr: float = 0.05):
        super().__init__(sens_present)
        self.lr = lr
        self.dual = None  # will be initialized per round length

    def compute_weights(self, local_meta: List[Dict[str, Any]]) -> List[float]:
        n = len(local_meta)
        if self.dual is None or len(self.dual) != n:
            self.dual = np.ones(n) / n
        # simplistic online ascent toward worst-client emphasis:
        # pretend factors inversely correlated to utility -> increase weights where factor is low
        factors = np.array([m["factor"] for m in local_meta])
        grad = -factors  # push weight to clients with low factor
        self.dual = np.maximum(self.dual + self.lr * grad, 1e-6)
        self.dual = self.dual / self.dual.sum()
        return self.dual.tolist()

def make_aggregator(sens_present: bool) -> BaseAggregator:
    return AFLAggregator(sens_present)
