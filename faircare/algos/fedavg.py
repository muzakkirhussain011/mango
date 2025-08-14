"""FedAvg implementation used by tests."""
from typing import Dict, List, Any

import torch

from faircare.algos.aggregator import BaseAggregator, register_aggregator


def _extract_num_samples(s: Dict[str, Any]) -> float:
    # Be permissive: accept several common keys seen across loaders/pipelines.
    for k in (
        "n_samples",
        "num_samples",
        "num_examples",
        "samples",
        "dataset_size",
        "train_size",
        "train_samples",
        "data_size",
        "size",
        "n",
    ):
        if k in s and s[k] is not None:
            try:
                return float(s[k])
            except Exception:
                pass
    # Nested hints (e.g., {"data": {"size": ...}})
    data = s.get("data")
    if isinstance(data, dict):
        for k in ("size", "n", "num_samples", "num_examples", "dataset_size"):
            if k in data and data[k] is not None:
                try:
                    return float(data[k])
                except Exception:
                    pass
    return 1.0


@register_aggregator("fedavg")
class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg).

    By default (as in McMahan et al.), the server aggregates client updates
    using weights proportional to each client's local dataset size.
    If the total size is zero / missing, we fall back to uniform.
    """
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        n = len(client_summaries)
        if n == 0:
            return torch.tensor([], dtype=torch.float32)

        sizes = torch.tensor([_extract_num_samples(s) for s in client_summaries], dtype=torch.float32)
        total = sizes.sum()
        if total > 0:
            weights = sizes / total
        else:
            weights = torch.ones(n, dtype=torch.float32) / n

        return self._postprocess(weights)
