"""FedProx implementation."""
from typing import List, Dict

import torch
from typing import List
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("fedprox")
class FedProxAggregator(BaseAggregator):
    """
    Federated Proximal (FedProx).
    
    Note: The proximal term is handled in client training.
    This aggregator is similar to FedAvg.
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks" (2020)
    """
    
    def __init__(
        self,
        n_clients: int,
        fedprox_mu: float = 0.01,
        **kwargs
    ):
        super().__init__(n_clients)
        self.mu = fedprox_mu
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """Weight by number of samples."""
        n_samples = torch.tensor([
            s.get("n_samples", 1) for s in client_summaries
        ], dtype=torch.float32)
        
        return n_samples / n_samples.sum()
