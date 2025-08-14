"""FedAvg implementation used by tests."""
from typing import Dict, List, Any

import torch

from faircare.algos.aggregator import BaseAggregator, register_aggregator


def _extract_num_samples(s: Dict[str, Any]) -> float:
    for k in ("n_samples", "num_samples", "samples", "dataset_size", "size", "n"):
        if k in s and s[k] is not None:
            try:
                return float(s[k])
            except Exception:
                pass
    return 1.0


@register_aggregator("fedavg")
class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg).
    - If `weighted=True`, weight ∝ client sample count.
    - Else uniform.
    """
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        n = len(client_summaries)
        if n == 0:
            return torch.tensor([], dtype=torch.float32)

        if self.weighted:
            sizes = torch.tensor([_extract_num_samples(s) for s in client_summaries], dtype=torch.float32)
            total = sizes.sum()
            weights = sizes / total if total > 0 else (torch.ones(n, dtype=torch.float32) / n)
        else:
            weights = torch.ones(n, dtype=torch.float32) / n

        return self._postprocess(weights)
