"""Agnostic Federated Learning (AFL) implementation."""

import torch
import numpy as np
from typing import List, Dict, Optional
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("afl")
class AFLAggregator(BaseAggregator):
    """
    Agnostic Federated Learning (AFL).
    
    Optimizes for worst-case performance across clients using
    a min-max optimization approach.
    
    Reference: Mohri et al., "Agnostic Federated Learning" (2019)
    """
    
    def __init__(
        self,
        n_clients: int,
        afl_lambda: float = 0.1,
        afl_smoothing: float = 0.01,
        **kwargs
    ):
        super().__init__(n_clients)
        self.lambda_param = afl_lambda
        self.smoothing = afl_smoothing
        
        # Initialize client weights uniformly
        self.client_weights = torch.ones(n_clients) / n_clients
        
        # Track losses for weight updates
        self.client_losses = torch.zeros(n_clients)
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """
        Compute AFL weights using exponential weighting.
        
        Reweight toward clients with worse performance.
        """
        # Extract losses
        selected_indices = []
        selected_losses = []
        
        for i, summary in enumerate(client_summaries):
            # Get client index if available
            client_id = summary.get("client_id", i)
            loss = summary.get("val_loss", summary.get("train_loss", 1.0))
            
            selected_indices.append(client_id)
            selected_losses.append(loss)
        
        # Update loss tracking
        for idx, loss in zip(selected_indices, selected_losses):
            self.client_losses[idx] = loss
        
        # Compute weights using exponential mechanism
        # Higher weight for higher loss (worse performance)
        exp_losses = torch.exp(self.lambda_param * self.client_losses)
        self.client_weights = exp_losses / exp_losses.sum()
        
        # Add smoothing
        self.client_weights = (
            (1 - self.smoothing) * self.client_weights +
            self.smoothing / self.n_clients
        )
        
        # Extract weights for selected clients
        weights = []
        for idx in selected_indices:
            weights.append(self.client_weights[idx].item())
        
        return torch.tensor(weights, dtype=torch.float32)
