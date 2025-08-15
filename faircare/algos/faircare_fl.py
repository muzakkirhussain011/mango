"""FairCare-FL: Our proposed fairness-aware federated learning algorithm."""
from typing import List, Dict, Any

import torch
import numpy as np

from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("faircare_fl")
class FairCareAggregator(BaseAggregator):
    """
    FairCare-FL: Novel fairness-aware aggregation with adaptive reweighting.
    
    Our proposed algorithm that achieves superior fairness-accuracy trade-off by:
    1. Dynamically adjusting client weights based on fairness metrics
    2. Using momentum to stabilize training
    3. Adaptive fairness penalty that decreases over rounds
    """
    
    def __init__(
        self,
        n_clients: int,
        alpha: float = 1.0,      # EO gap weight
        beta: float = 0.5,       # FPR gap weight  
        gamma: float = 0.5,      # SP gap weight
        delta: float = 0.1,      # Accuracy weight
        tau: float = 1.0,        # Temperature for softmax
        mu: float = 0.9,         # Momentum coefficient
        epsilon: float = 0.01,   # Weight floor
        tau_anneal: bool = True, # Anneal temperature over rounds
        weight_clip: float = 10.0,
        fairness_metric: str = "eo_gap",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip, fairness_metric=fairness_metric)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tau = tau
        self.initial_tau = tau
        self.mu = mu
        self.tau_anneal = tau_anneal
        self.round_num = 0
        
        # Track historical performance for momentum
        self.client_history = {}
        for i in range(n_clients):
            self.client_history[i] = {
                'fairness_score': 0.5,  # Initialize to neutral
                'accuracy': 0.5,
                'weight': 1.0 / n_clients
            }
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute FairCare-FL weights with improved balancing.
        
        Key innovations:
        1. Combine multiple fairness metrics with accuracy
        2. Use historical performance with momentum
        3. Adaptive temperature annealing
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Anneal temperature over rounds for better convergence
        if self.tau_anneal and self.round_num > 1:
            self.tau = self.initial_tau / (1 + 0.1 * self.round_num)
        
        scores = []
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            
            # Extract metrics (with defaults to handle missing values)
            eo_gap = summary.get("eo_gap", summary.get("val_EO_gap", 0.5))
            fpr_gap = summary.get("fpr_gap", summary.get("val_FPR_gap", 0.5))
            sp_gap = summary.get("sp_gap", summary.get("val_SP_gap", 0.5))
            
            # Get accuracy or use loss as proxy
            if "val_acc" in summary:
                accuracy = summary["val_acc"]
            elif "val_accuracy" in summary:
                accuracy = summary["val_accuracy"]
            elif "val_loss" in summary:
                # Convert loss to accuracy proxy (lower loss = higher accuracy)
                accuracy = 1.0 / (1.0 + summary["val_loss"])
            else:
                accuracy = 0.5  # Default
            
            # Compute fairness score (lower gaps are better)
            # Normalize gaps to [0, 1] range
            fairness_score = (
                self.alpha * (1.0 - min(eo_gap, 1.0)) +
                self.beta * (1.0 - min(fpr_gap, 1.0)) +
                self.gamma * (1.0 - min(abs(sp_gap), 1.0))
            )
            fairness_score = fairness_score / (self.alpha + self.beta + self.gamma)
            
            # Combine fairness and accuracy
            # Higher weight for better fairness AND good accuracy
            combined_score = (1 - self.delta) * fairness_score + self.delta * accuracy
            
            # Apply momentum from historical performance
            if client_id in self.client_history:
                old_score = self.client_history[client_id]['fairness_score']
                combined_score = self.mu * old_score + (1 - self.mu) * combined_score
                
                # Update history
                self.client_history[client_id]['fairness_score'] = combined_score
                self.client_history[client_id]['accuracy'] = accuracy
            
            scores.append(combined_score)
        
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Apply softmax with temperature
        # Higher scores get higher weights
        if self.tau > 0:
            weights = torch.softmax(scores / self.tau, dim=0)
        else:
            # If tau is 0, assign all weight to best client
            weights = torch.zeros_like(scores)
            weights[torch.argmax(scores)] = 1.0
        
        # Apply weight floor and clipping
        weights = self._postprocess(weights)
        
        # Store weights in history for analysis
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            if client_id in self.client_history:
                self.client_history[client_id]['weight'] = weights[i].item()
        
        return weights
    
    def get_client_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics about client performance and weights."""
        return self.client_history.copy()
