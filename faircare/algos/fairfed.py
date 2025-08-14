"""FairFed: inverse-gap client reweighting with floors/clipping."""
from typing import List, Dict, Any

import torch

from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("fairfed")
class FairFedAggregator(BaseAggregator):
    """
    Simple fairness-aware weighting: lower gap ⇒ higher weight.
    Uses `fairness_metric` key from client summaries (default: 'eo_gap').
    """
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        gaps = []
        for s in client_summaries:
            val = s.get(self.fairness_metric, None)
            if val is None:
                val = s.get("val_loss", 0.0)
            gaps.append(float(val) + 1e-6)  # avoid division by zero

        g = torch.tensor(gaps, dtype=torch.float32)
        inv = 1.0 / g
        weights = inv / inv.sum()
        return self._postprocess(weights)
