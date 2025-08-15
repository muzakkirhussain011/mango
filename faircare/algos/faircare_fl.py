"""Enhanced FairCare-FL: Superior fairness-aware federated learning algorithm."""
from typing import List, Dict, Any
import torch
import numpy as np
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("faircare_fl")
class FairCareAggregator(BaseAggregator):
    """
    Enhanced FairCare-FL: Achieves superior fairness-accuracy trade-off.
    
    Key innovations:
    1. Dual-objective optimization with adaptive balancing
    2. Client performance tracking with variance penalties
    3. Sophisticated momentum with selective application
    4. Dynamic fairness-accuracy trade-off based on convergence
    """
    
    def __init__(
        self,
        n_clients: int,
        # Fairness weights (increased for better fairness)
        alpha: float = 2.0,      # EO gap weight (increased)
        beta: float = 1.0,       # FPR gap weight (increased)
        gamma: float = 1.0,      # SP gap weight (increased)
        delta: float = 0.05,     # Accuracy weight (decreased for more fairness focus)
        # Temperature and momentum
        tau: float = 0.8,        # Lower initial temperature for more decisive weighting
        mu: float = 0.7,         # Reduced momentum for better adaptability
        # Weight constraints
        epsilon: float = 0.05,   # Higher floor to ensure participation
        weight_clip: float = 5.0, # Tighter clipping to prevent dominance
        # Advanced parameters
        tau_anneal: bool = True,
        variance_penalty: float = 0.3,  # Penalty for high variance clients
        boost_factor: float = 1.5,      # Boost for consistently fair clients
        convergence_threshold: float = 0.01,  # For adaptive trade-off
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
        self.variance_penalty = variance_penalty
        self.boost_factor = boost_factor
        self.convergence_threshold = convergence_threshold
        self.round_num = 0
        
        # Enhanced tracking for each client
        self.client_history = {}
        for i in range(n_clients):
            self.client_history[i] = {
                'fairness_scores': [],  # Track history of scores
                'accuracy_scores': [],
                'eo_gaps': [],
                'weights': [],
                'participation_count': 0,
                'improvement_rate': 0.0,
                'stability_score': 1.0,
            }
        
        # Global convergence tracking
        self.global_fairness_trend = []
        self.is_converging = False
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Enhanced weight computation with sophisticated fairness-aware strategy.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Adaptive temperature annealing (slower than original)
        if self.tau_anneal and self.round_num > 1:
            # Use exponential decay instead of linear
            self.tau = self.initial_tau * np.exp(-0.05 * self.round_num)
            self.tau = max(self.tau, 0.1)  # Keep minimum temperature
        
        # Adaptive trade-off based on convergence
        if self.round_num > 5:
            self._update_convergence_status()
            if self.is_converging:
                # Once converging, focus more on fairness
                self.delta = max(0.01, self.delta * 0.9)
        
        scores = []
        client_indices = []
        
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            client_indices.append(client_id)
            
            # Extract all available metrics with smart defaults
            eo_gap = self._get_metric(summary, ["eo_gap", "val_EO_gap", "EO_gap"], 0.3)
            fpr_gap = self._get_metric(summary, ["fpr_gap", "val_FPR_gap", "FPR_gap"], 0.3)
            sp_gap = self._get_metric(summary, ["sp_gap", "val_SP_gap", "SP_gap"], 0.3)
            accuracy = self._get_accuracy(summary)
            
            # Update client history
            if client_id in self.client_history:
                hist = self.client_history[client_id]
                hist['eo_gaps'].append(eo_gap)
                hist['accuracy_scores'].append(accuracy)
                hist['participation_count'] += 1
                
                # Calculate improvement rate (positive means getting better)
                if len(hist['eo_gaps']) > 1:
                    recent_improvement = hist['eo_gaps'][-2] - hist['eo_gaps'][-1]
                    hist['improvement_rate'] = 0.7 * hist['improvement_rate'] + 0.3 * recent_improvement
                
                # Calculate stability (lower variance is better)
                if len(hist['eo_gaps']) > 2:
                    variance = np.var(hist['eo_gaps'][-3:])
                    hist['stability_score'] = 1.0 / (1.0 + variance)
            
            # Compute enhanced fairness score
            fairness_score = self._compute_enhanced_fairness_score(
                eo_gap, fpr_gap, sp_gap, client_id
            )
            
            # Combine with accuracy using adaptive weighting
            combined_score = (1 - self.delta) * fairness_score + self.delta * accuracy
            
            # Apply bonuses and penalties
            combined_score = self._apply_adjustments(combined_score, client_id)
            
            # Store final score
            if client_id in self.client_history:
                self.client_history[client_id]['fairness_scores'].append(combined_score)
            
            scores.append(combined_score)
        
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Enhanced weight computation using softmax with temperature
        if self.tau > 0:
            # Normalize scores to prevent overflow
            scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-8)
            
            # Apply softmax with temperature
            exp_scores = torch.exp(scores_normalized / self.tau)
            weights = exp_scores / exp_scores.sum()
            
            # Apply selective momentum for stable clients
            weights = self._apply_selective_momentum(weights, client_indices)
        else:
            # Deterministic: assign all weight to best client
            weights = torch.zeros_like(scores)
            weights[torch.argmax(scores)] = 1.0
        
        # Apply constraints (floor and ceiling)
        weights = self._postprocess(weights)
        
        # Store weights in history
        for i, client_id in enumerate(client_indices):
            if client_id in self.client_history:
                self.client_history[client_id]['weights'].append(weights[i].item())
        
        # Update global fairness trend
        avg_eo_gap = np.mean([self.client_history[cid]['eo_gaps'][-1] 
                              for cid in client_indices 
                              if cid in self.client_history and self.client_history[cid]['eo_gaps']])
        self.global_fairness_trend.append(avg_eo_gap)
        
        return weights
    
    def _get_metric(self, summary: Dict, keys: List[str], default: float) -> float:
        """Extract metric from summary with fallback keys."""
        for key in keys:
            if key in summary:
                return float(summary[key])
        return default
    
    def _get_accuracy(self, summary: Dict) -> float:
        """Extract accuracy or estimate from loss."""
        if "val_acc" in summary:
            return summary["val_acc"]
        elif "val_accuracy" in summary:
            return summary["val_accuracy"]
        elif "accuracy" in summary:
            return summary["accuracy"]
        elif "val_loss" in summary:
            # Better loss-to-accuracy conversion
            loss = summary["val_loss"]
            return 1.0 / (1.0 + loss)
        elif "train_loss" in summary:
            loss = summary["train_loss"]
            return 1.0 / (1.0 + 1.5 * loss)  # Penalize train loss more
        else:
            return 0.5
    
    def _compute_enhanced_fairness_score(
        self, eo_gap: float, fpr_gap: float, sp_gap: float, client_id: int
    ) -> float:
        """
        Compute sophisticated fairness score with non-linear penalties.
        """
        # Apply non-linear transformation to gaps (square root for diminishing penalties)
        eo_penalty = np.sqrt(min(eo_gap, 1.0))
        fpr_penalty = np.sqrt(min(fpr_gap, 1.0))
        sp_penalty = np.sqrt(min(abs(sp_gap), 1.0))
        
        # Weighted combination with normalization
        total_weight = self.alpha + self.beta + self.gamma
        fairness_score = (
            self.alpha * (1.0 - eo_penalty) +
            self.beta * (1.0 - fpr_penalty) +
            self.gamma * (1.0 - sp_penalty)
        ) / total_weight
        
        # Apply historical context if available
        if client_id in self.client_history:
            hist = self.client_history[client_id]
            
            # Boost if improving
            if hist['improvement_rate'] > 0:
                fairness_score *= (1.0 + 0.2 * min(hist['improvement_rate'], 1.0))
            
            # Boost if stable
            fairness_score *= (0.9 + 0.1 * hist['stability_score'])
        
        return min(max(fairness_score, 0.0), 1.0)
    
    def _apply_adjustments(self, score: float, client_id: int) -> float:
        """Apply bonuses and penalties based on client history."""
        if client_id not in self.client_history:
            return score
        
        hist = self.client_history[client_id]
        
        # Participation bonus (encourage diverse participation)
        if hist['participation_count'] < 3:
            score *= 1.1
        
        # Consistency bonus
        if len(hist['fairness_scores']) > 2:
            recent_scores = hist['fairness_scores'][-3:]
            variance = np.var(recent_scores)
            if variance < 0.01:  # Very consistent
                score *= self.boost_factor
            elif variance > 0.1:  # High variance penalty
                score *= (1.0 - self.variance_penalty * min(variance, 1.0))
        
        # Top performer bonus
        if len(hist['eo_gaps']) > 0:
            latest_gap = hist['eo_gaps'][-1]
            if latest_gap < 0.1:  # Excellent fairness
                score *= 1.3
        
        return min(max(score, 0.0), 2.0)  # Cap adjustments
    
    def _apply_selective_momentum(
        self, weights: torch.Tensor, client_indices: List[int]
    ) -> torch.Tensor:
        """Apply momentum only to stable clients."""
        if self.round_num <= 2:
            return weights
        
        momentum_weights = weights.clone()
        
        for i, client_id in enumerate(client_indices):
            if client_id in self.client_history:
                hist = self.client_history[client_id]
                
                # Only apply momentum if client is stable and has history
                if len(hist['weights']) > 0 and hist['stability_score'] > 0.7:
                    old_weight = hist['weights'][-1]
                    # Adaptive momentum based on stability
                    adaptive_mu = self.mu * hist['stability_score']
                    momentum_weights[i] = adaptive_mu * old_weight + (1 - adaptive_mu) * weights[i]
        
        # Renormalize
        return momentum_weights / momentum_weights.sum()
    
    def _update_convergence_status(self):
        """Check if training is converging based on fairness trends."""
        if len(self.global_fairness_trend) < 5:
            return
        
        recent_trend = self.global_fairness_trend[-5:]
        variance = np.var(recent_trend)
        
        # Consider converged if variance is low
        self.is_converging = variance < self.convergence_threshold
    
    def get_client_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed statistics about client performance."""
        stats = {}
        for client_id, hist in self.client_history.items():
            stats[client_id] = {
                'participation_count': hist['participation_count'],
                'avg_eo_gap': np.mean(hist['eo_gaps']) if hist['eo_gaps'] else None,
                'avg_weight': np.mean(hist['weights']) if hist['weights'] else None,
                'improvement_rate': hist['improvement_rate'],
                'stability_score': hist['stability_score'],
                'latest_fairness_score': hist['fairness_scores'][-1] if hist['fairness_scores'] else None
            }
        return stats
