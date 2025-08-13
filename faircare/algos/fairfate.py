"""FAIR-FATE implementation."""

import torch
import numpy as np
from typing import List, Dict, Optional
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("fairfate")
class FairFateAggregator(BaseAggregator):
    """
    FAIR-FATE: Fairness-aware federated learning.
    
    Uses server-side fairness metrics to guide aggregation.
    
    Reference: Ezzeldin et al., "FairFed: Enabling Group Fairness
    in Federated Learning" (2021)
    """
    
    def __init__(
        self,
        n_clients: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.1,
        tau: float = 1.0,
        epsilon: float = 0.01,
        **kwargs
    ):
        super().__init__(n_clients)
        self.alpha = alpha  # EO gap weight
        self.beta = beta    # FPR gap weight
        self.gamma = gamma  # SP gap weight
        self.delta = delta  # val loss weight
        self.tau = tau      # temperature
        self.epsilon = epsilon  # weight floor
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """
        Compute fairness-aware weights.
        
        Lower fairness score = higher weight.
        """
        scores = []
        
        for summary in client_summaries:
            # Extract fairness metrics
            eo_gap = summary.get("eo_gap", 0.0)
            fpr_gap = summary.get("fpr_gap", 0.0)
            sp_gap = summary.get("sp_gap", 0.0)
            val_loss = summary.get("val_loss", 1.0)
            
            # Compute fairness score (lower is better)
            score = (
                self.alpha * eo_gap +
                self.beta * fpr_gap +
                self.gamma * abs(sp_gap) +
                self.delta * val_loss
            )
            scores.append(score)
        
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Convert to weights using softmin
        # w_k = exp(-s_k/τ) / Σ_j exp(-s_j/τ)
        weights = torch.exp(-scores / self.tau)
        weights = weights / weights.sum()
        
        # Apply weight floor
        weights = torch.maximum(weights, torch.tensor(self.epsilon))
        weights = weights / weights.sum()
        
        return weights
