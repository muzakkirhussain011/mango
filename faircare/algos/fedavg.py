"""FedAvg implementation used by tests."""
from typing import Dict, List, Any

import torch

from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("fedavg")
class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg).
    Computes either uniform weights or dataset-size weighted averaging.
    """
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        if self.weighted:
            n_samples = torch.tensor([s.get("n_samples", 1) for s in client_summaries], dtype=torch.float32)
            weights = n_samples / n_samples.sum() if n_samples.sum() > 0 else torch.ones(len(client_summaries)) / len(client_summaries)
        else:
            n = len(client_summaries)
            weights = torch.ones(n, dtype=torch.float32) / max(n, 1)

        return self._postprocess(weights)
