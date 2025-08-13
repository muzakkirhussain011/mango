"""FedAvg implementation."""

import torch
from typing import List, Dict
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("fedavg")
class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg).
    
    Reference: McMahan et al., "Communication-Efficient Learning of
    Deep Networks from Decentralized Data" (2017)
    """
    
    def __init__(
        self,
        n_clients: int,
        weighted: bool = True,
        **kwargs
    ):
        super().__init__(n_clients)
        self.weighted = weighted
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """
        Compute weights based on number of samples.
        
        Args:
            client_summaries: List of client statistics
        
        Returns:
            Aggregation weights
        """
        if self.weighted:
            # Weight by number of samples
            n_samples = torch.tensor([
                s.get("n_samples", 1) for s in client_summaries
            ], dtype=torch.float32)
            weights = n_samples / n_samples.sum()
        else:
            # Uniform weights
            n = len(client_summaries)
            weights = torch.ones(n) / n
        
        return weights
