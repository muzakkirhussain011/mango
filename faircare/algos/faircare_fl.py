# faircare/algos/faircare_fl.py
"""FairCare-FL: Our proposed fairness-aware federated learning algorithm."""
from typing import List, Dict, Optional

import torch
import numpy as np
from typing import List
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("faircare_fl")
class FairCareAggregator(BaseAggregator):
    """
    FairCare-FL: Fairness-aware federated learning with momentum.
    
    Server-side fairness-aware aggregation with momentum, inspired by
    FAIR-FATE but generalized with validation-set fairness steering.
    """
    
    def __init__(
        self,
        n_clients: int,
        alpha: float = 1.0,     # EO gap weight
        beta: float = 0.5,      # FPR gap weight  
        gamma: float = 0.5,     # SP gap weight
        delta: float = 0.1,     # val loss weight
        tau: float = 1.0,       # temperature
        mu: float = 0.9,        # momentum coefficient
        epsilon: float = 0.01,  # weight floor
        faircare_momentum: float = 0.9,
        tau_anneal: bool = False,
        faircare_anneal_rounds: int = 5,
        weight_clip: float = 10.0,
        **kwargs
    ):
        super().__init__(n_clients)
        
        # Fairness score weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Aggregation parameters
        self.tau = tau
        self.initial_tau = tau
        self.mu = faircare_momentum if faircare_momentum is not None else mu
        self.epsilon = epsilon
        self.weight_clip = weight_clip
        
        # Annealing
        self.tau_anneal = tau_anneal
        self.anneal_rounds = faircare_anneal_rounds
        
        # Enable momentum
        self.use_momentum = True
        self.momentum = self.mu
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """
        Compute FairCare-FL weights based on fairness scores.
        
        Implements the pseudocode:
        score s_k = α*eo_gap + β*fpr_gap + γ*|sp_gap| + δ*val_loss
        w_k_raw = exp(-s_k/τ) / Σ_j exp(-s_j/τ)
        w_k = max(w_k_raw, ε); normalize
        """
        scores = []
        
        for summary in client_summaries:
            # Extract metrics from client summary
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
        
        # Temperature annealing
        if self.tau_anneal and self.round < self.anneal_rounds:
            # Gradually decrease temperature
            progress = self.round / self.anneal_rounds
            self.tau = self.initial_tau * (1 - 0.5 * progress)
        
        # Softmin weighting: w_k = exp(-s_k/τ) / Σ exp(-s_j/τ)
        weights = torch.exp(-scores / self.tau)
        weights = weights / weights.sum()
        
        # Apply weight floor
        weights = torch.maximum(weights, torch.tensor(self.epsilon))
        
        # Weight clipping for stability
        if self.weight_clip > 0:
            max_weight = 1.0 / len(weights) * self.weight_clip
            weights = torch.minimum(weights, torch.tensor(max_weight))
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
