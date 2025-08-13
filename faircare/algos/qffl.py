"""q-FFL (q-Fair Federated Learning) implementation."""

import torch
import numpy as np
from typing import List, Dict
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("qffl")
class QFFLAggregator(BaseAggregator):
    """
    q-Fair Federated Learning (q-FFL).
    
    Optimizes for uniform performance across clients by reweighting
    based on loss magnitudes.
    
    Reference: Li et al., "Fair Resource Allocation in Federated Learning" (2019)
    """
    
    def __init__(
        self,
        n_clients: int,
        q: float = 2.0,
        q_eps: float = 1e-4,
        **kwargs
    ):
        super().__init__(n_clients)
        self.q = q
        self.eps = q_eps
        
        # Track client losses
        self.client_losses = {}
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """
        Compute q-FFL weights.
        
        Weight = (∂/∂θ) of ∑_k (ℓ_k)^q / q
        This gives higher weight to clients with higher loss.
        """
        losses = []
        for i, summary in enumerate(client_summaries):
            loss = summary.get("train_loss", 1.0)
            losses.append(loss)
            self.client_losses[i] = loss
        
        losses = torch.tensor(losses, dtype=torch.float32)
        
        # Compute q-FFL weights
        if self.q == 1.0:
            # Standard FedAvg
            weights = torch.ones_like(losses)
        else:
            # q-FFL weighting: w_k ∝ (loss_k)^(q-1)
            weights = torch.pow(losses + self.eps, self.q - 1)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
