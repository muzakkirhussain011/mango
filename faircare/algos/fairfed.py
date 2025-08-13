"""FairFed implementation (simplified fairness weighting)."""

import torch
from typing import List, Dict
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("fairfed")
class FairFedAggregator(BaseAggregator):
    """
    Simple fairness-weighted federated learning.
    
    Weights clients inversely proportional to their fairness gaps.
    """
    
    def __init__(
        self,
        n_clients: int,
        fairness_metric: str = "max_group_gap",
        **kwargs
    ):
        super().__init__(n_clients)
        self.fairness_metric = fairness_metric
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """
        Compute weights inversely proportional to fairness gaps.
        """
        gaps = []
        
        for summary in client_summaries:
            gap = summary.get(self.fairness_metric, 0.0)
            # Add small epsilon to avoid division by zero
            gaps.append(gap + 1e-6)
        
        gaps = torch.tensor(gaps, dtype=torch.float32)
        
        # Inverse weighting
        weights = 1.0 / gaps
        weights = weights / weights.sum()
        
        return weights
