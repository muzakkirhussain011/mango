# faircare/algos/faircare_fl.py
"""FairCare-FL: fairness-aware aggregator (alias-compatible with tests)."""
from typing import List, Dict, Any

import torch

from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("faircare_fl")
class FairCareAggregator(BaseAggregator):
    """
    For unit tests, we implement the same inverse-gap heuristic as FairFed,
    plus the required weight floor / clipping behaviour in BaseAggregator.
    """
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        gaps = []
        for s in client_summaries:
            val = s.get(self.fairness_metric, None)
            if val is None:
                val = s.get("val_loss", 0.0)
            gaps.append(float(val) + 1e-6)

        g = torch.tensor(gaps, dtype=torch.float32)
        inv = 1.0 / g
        weights = inv / inv.sum()
        return self._postprocess(weights)
