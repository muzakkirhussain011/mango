"""FairCare-FL++: Next-generation fair federated learning algorithm."""
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("faircare_fl")
class FairCarePlusPlusAggregator(BaseAggregator):
    """
    FairCare-FL++: Dual-level fairness optimization with adaptive bias mitigation.
    
    Key innovations:
    1. Dual-level fairness optimization (local training + global aggregation)
    2. Adaptive fairness weighting with bias detection
    3. Fairness-aware momentum at both client and server levels
    4. Multi-metric fairness support
    5. Automatic bias mitigation mode
    """
    
    def __init__(
        self,
        n_clients: int,
        # Fairness metric weights
        alpha: float = 2.0,      # EO gap weight
        beta: float = 1.5,       # FPR gap weight  
        gamma: float = 1.5,      # SP gap weight
        delta: float = 0.1,      # Accuracy weight
        # Adaptive parameters
        tau: float = 1.0,        # Initial temperature
        tau_min: float = 0.1,    # Minimum temperature
        tau_anneal_rate: float = 0.95,  # Temperature annealing rate
        mu_client: float = 0.9,  # Client-side momentum
        theta_server: float = 0.8,  # Server-side momentum
        # Bias detection thresholds
        bias_threshold_eo: float = 0.15,  # EO gap threshold for bias mode
        bias_threshold_fpr: float = 0.15,  # FPR gap threshold
        bias_threshold_sp: float = 0.2,   # SP gap threshold
        # Fairness penalty for local training
        lambda_fair: float = 0.5,  # Initial fairness penalty weight
        lambda_min: float = 0.1,   # Minimum lambda
        lambda_max: float = 2.0,   # Maximum lambda
        lambda_adapt_rate: float = 1.2,  # Lambda adjustment rate
        # Weight constraints
        epsilon: float = 0.01,   # Weight floor
        weight_clip: float = 10.0,  # Weight ceiling multiplier
        # Advanced features
        enable_bias_detection: bool = True,
        enable_server_momentum: bool = True,
        enable_multi_metric: bool = True,
        variance_penalty: float = 0.2,
        improvement_bonus: float = 0.3,
        fairness_metric: str = "composite",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip, fairness_metric=fairness_metric)
        
        # Core parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Temperature parameters
        self.tau = tau
        self.tau_initial = tau
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        
        # Momentum parameters
        self.mu_client = mu_client
        self.theta_server = theta_server
        
        # Bias detection parameters
        self.bias_threshold_eo = bias_threshold_eo
        self.bias_threshold_fpr = bias_threshold_fpr
        self.bias_threshold_sp = bias_threshold_sp
        self.enable_bias_detection = enable_bias_detection
        
        # Fairness penalty parameters
        self.lambda_fair = lambda_fair
        self.lambda_initial = lambda_fair
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_adapt_rate = lambda_adapt_rate
        
        # Advanced features
        self.enable_server_momentum = enable_server_momentum
        self.enable_multi_metric = enable_multi_metric
        self.variance_penalty = variance_penalty
        self.improvement_bonus = improvement_bonus
        
        # State tracking
        self.round_num = 0
        self.bias_mitigation_mode = False
        self.consecutive_bias_rounds = 0
        self.server_momentum_buffer = None
        
        # Client history with enhanced tracking
        self.client_history = {}
        for i in range(n_clients):
            self.client_history[i] = {
                'fairness_scores': [],
                'smoothed_score': 0.5,
                'eo_gaps': [],
                'fpr_gaps': [],
                'sp_gaps': [],
                'accuracies': [],
                'weights': [],
                'participation_count': 0,
                'improvement_trend': 0.0,
                'stability_score': 1.0,
                'last_update_quality': 0.5
            }
        
        # Global metrics tracking
        self.global_metrics_history = {
            'eo_gap': [],
            'fpr_gap': [],
            'sp_gap': [],
            'accuracy': [],
            'worst_group_f1': []
        }
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute adaptive weights with bias detection and mitigation.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Detect and respond to bias
        if self.enable_bias_detection:
            self._detect_and_respond_to_bias(client_summaries)
        
        # Adaptive temperature annealing
        if self.round_num > 1:
            if self.bias_mitigation_mode:
                # Sharp temperature for aggressive bias mitigation
                self.tau = max(self.tau_min, self.tau_initial * 0.5)
            else:
                # Normal annealing
                self.tau = max(self.tau_min, self.tau * self.tau_anneal_rate)
        
        scores = []
        client_indices = []
        
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            client_indices.append(client_id)
            
            # Extract comprehensive metrics
            metrics = self._extract_metrics(summary)
            
            # Update client history
            self._update_client_history(client_id, metrics)
            
            # Compute composite fairness score
            if self.enable_multi_metric:
                fairness_score = self._compute_multi_metric_score(metrics, client_id)
            else:
                fairness_score = self._compute_basic_fairness_score(metrics)
            
            # Apply client-side momentum
            if client_id in self.client_history:
                hist = self.client_history[client_id]
                smoothed_score = self.mu_client * hist['smoothed_score'] + (1 - self.mu_client) * fairness_score
                hist['smoothed_score'] = smoothed_score
                fairness_score = smoothed_score
            
            # Apply performance-based adjustments
            fairness_score = self._apply_performance_adjustments(fairness_score, client_id)
            
            scores.append(fairness_score)
        
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Convert scores to weights using softmax with temperature
        if self.tau > 0:
            # Normalize scores for numerical stability
            scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-8)
            exp_scores = torch.exp(scores_normalized / self.tau)
            weights = exp_scores / exp_scores.sum()
        else:
            # Deterministic: best client gets all weight
            weights = torch.zeros_like(scores)
            weights[torch.argmax(scores)] = 1.0
        
        # Apply weight constraints
        weights = self._postprocess(weights)
        
        # Store weights in history
        for i, client_id in enumerate(client_indices):
            if client_id in self.client_history:
                self.client_history[client_id]['weights'].append(weights[i].item())
        
        # Update global metrics
        self._update_global_metrics(client_summaries)
        
        return weights
    
    def _extract_metrics(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Extract all relevant metrics from client summary."""
        metrics = {}
        
        # Fairness gaps
        metrics['eo_gap'] = summary.get('eo_gap', summary.get('val_EO_gap', 0.3))
        metrics['fpr_gap'] = summary.get('fpr_gap', summary.get('val_FPR_gap', 0.3))
        metrics['sp_gap'] = summary.get('sp_gap', summary.get('val_SP_gap', 0.3))
        
        # Performance metrics
        metrics['accuracy'] = summary.get('val_acc', summary.get('val_accuracy', 0.5))
        metrics['loss'] = summary.get('val_loss', summary.get('train_loss', 1.0))
        
        # Additional fairness metrics if available
        metrics['worst_group_f1'] = summary.get('worst_group_F1', 0.5)
        metrics['max_gap'] = summary.get('max_group_gap', max(metrics['eo_gap'], metrics['fpr_gap'], abs(metrics['sp_gap'])))
        
        return metrics
    
    def _update_client_history(self, client_id: int, metrics: Dict[str, float]):
        """Update client history with new metrics."""
        if client_id not in self.client_history:
            return
        
        hist = self.client_history[client_id]
        
        # Store metrics
        hist['eo_gaps'].append(metrics['eo_gap'])
        hist['fpr_gaps'].append(metrics['fpr_gap'])
        hist['sp_gaps'].append(metrics['sp_gap'])
        hist['accuracies'].append(metrics['accuracy'])
        hist['participation_count'] += 1
        
        # Calculate improvement trend
        if len(hist['eo_gaps']) > 1:
            recent_improvement = 0
            for gap_list in [hist['eo_gaps'], hist['fpr_gaps'], hist['sp_gaps']]:
                if len(gap_list) > 1:
                    recent_improvement += (gap_list[-2] - gap_list[-1])
            hist['improvement_trend'] = 0.7 * hist['improvement_trend'] + 0.3 * recent_improvement / 3
        
        # Calculate stability score
        if len(hist['eo_gaps']) > 2:
            recent_gaps = hist['eo_gaps'][-3:]
            variance = np.var(recent_gaps)
            hist['stability_score'] = 1.0 / (1.0 + 10 * variance)  # More sensitive to variance
        
        # Assess update quality
        quality = (1 - metrics['eo_gap']) * (1 - metrics['fpr_gap']) * (1 - abs(metrics['sp_gap'])) * metrics['accuracy']
        hist['last_update_quality'] = quality
    
    def _compute_multi_metric_score(self, metrics: Dict[str, float], client_id: int) -> float:
        """Compute sophisticated multi-metric fairness score."""
        # Non-linear transformation of gaps (square root for diminishing returns)
        eo_score = 1.0 - np.sqrt(min(metrics['eo_gap'], 1.0))
        fpr_score = 1.0 - np.sqrt(min(metrics['fpr_gap'], 1.0))
        sp_score = 1.0 - np.sqrt(min(abs(metrics['sp_gap']), 1.0))
        
        # Weighted combination
        fairness_component = (
            self.alpha * eo_score +
            self.beta * fpr_score +
            self.gamma * sp_score
        ) / (self.alpha + self.beta + self.gamma)
        
        # Include worst group F1 if available
        if metrics['worst_group_f1'] > 0:
            fairness_component = 0.8 * fairness_component + 0.2 * metrics['worst_group_f1']
        
        # Combine with accuracy
        if self.bias_mitigation_mode:
            # Heavily prioritize fairness in bias mitigation mode
            combined_score = 0.95 * fairness_component + 0.05 * metrics['accuracy']
        else:
            combined_score = (1 - self.delta) * fairness_component + self.delta * metrics['accuracy']
        
        return min(max(combined_score, 0.0), 1.0)
    
    def _compute_basic_fairness_score(self, metrics: Dict[str, float]) -> float:
        """Compute basic fairness score."""
        fairness = (
            self.alpha * (1 - metrics['eo_gap']) +
            self.beta * (1 - metrics['fpr_gap']) +
            self.gamma * (1 - abs(metrics['sp_gap']))
        ) / (self.alpha + self.beta + self.gamma)
        
        return (1 - self.delta) * fairness + self.delta * metrics['accuracy']
    
    def _apply_performance_adjustments(self, score: float, client_id: int) -> float:
        """Apply bonuses and penalties based on performance history."""
        if client_id not in self.client_history:
            return score
        
        hist = self.client_history[client_id]
        
        # Improvement bonus
        if hist['improvement_trend'] > 0.01:
            score *= (1.0 + self.improvement_bonus * min(hist['improvement_trend'], 1.0))
        
        # Stability bonus/penalty
        if hist['stability_score'] > 0.8:
            score *= 1.1  # Stable client bonus
        elif hist['stability_score'] < 0.3:
            score *= (1.0 - self.variance_penalty)  # Unstable client penalty
        
        # Quality bonus for consistently good updates
        if hist['last_update_quality'] > 0.7:
            score *= 1.2
        
        # Participation encouragement for new clients
        if hist['participation_count'] <= 2:
            score *= 1.15
        
        return min(max(score, 0.0), 2.0)
    
    def _detect_and_respond_to_bias(self, client_summaries: List[Dict[str, Any]]):
        """Detect bias and adjust algorithm parameters."""
        # Calculate average metrics across clients
        avg_eo = np.mean([self._extract_metrics(s)['eo_gap'] for s in client_summaries])
        avg_fpr = np.mean([self._extract_metrics(s)['fpr_gap'] for s in client_summaries])
        avg_sp = np.mean([abs(self._extract_metrics(s)['sp_gap']) for s in client_summaries])
        
        # Check if bias exceeds thresholds
        bias_detected = (
            avg_eo > self.bias_threshold_eo or
            avg_fpr > self.bias_threshold_fpr or
            avg_sp > self.bias_threshold_sp
        )
        
        if bias_detected:
            if not self.bias_mitigation_mode:
                # Enter bias mitigation mode
                self.bias_mitigation_mode = True
                self.consecutive_bias_rounds = 1
                
                # Increase fairness penalty for local training
                self.lambda_fair = min(self.lambda_max, self.lambda_fair * self.lambda_adapt_rate)
                
                # Adjust aggregator parameters for stronger fairness focus
                self.delta = max(0.01, self.delta * 0.5)  # Reduce accuracy weight
                
            else:
                self.consecutive_bias_rounds += 1
                
                # Further increase fairness penalty if bias persists
                if self.consecutive_bias_rounds > 2:
                    self.lambda_fair = min(self.lambda_max, self.lambda_fair * 1.1)
        else:
            if self.bias_mitigation_mode:
                # Exit bias mitigation mode
                self.bias_mitigation_mode = False
                self.consecutive_bias_rounds = 0
                
                # Gradually relax fairness penalty
                self.lambda_fair = max(self.lambda_min, self.lambda_fair * 0.9)
                
                # Restore balance between fairness and accuracy
                self.delta = min(0.2, self.delta * 1.5)
    
    def _update_global_metrics(self, client_summaries: List[Dict[str, Any]]):
        """Update global metrics history."""
        if not client_summaries:
            return
        
        # Calculate average metrics
        metrics = [self._extract_metrics(s) for s in client_summaries]
        
        self.global_metrics_history['eo_gap'].append(np.mean([m['eo_gap'] for m in metrics]))
        self.global_metrics_history['fpr_gap'].append(np.mean([m['fpr_gap'] for m in metrics]))
        self.global_metrics_history['sp_gap'].append(np.mean([abs(m['sp_gap']) for m in metrics]))
        self.global_metrics_history['accuracy'].append(np.mean([m['accuracy'] for m in metrics]))
        
        if any('worst_group_f1' in m for m in metrics):
            wgf1_values = [m['worst_group_f1'] for m in metrics if 'worst_group_f1' in m]
            self.global_metrics_history['worst_group_f1'].append(np.mean(wgf1_values))
    
    def get_fairness_config(self) -> Dict[str, Any]:
        """Get current fairness configuration for clients."""
        return {
            'lambda_fair': self.lambda_fair,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'round': self.round_num,
            'target_metrics': {
                'eo_gap': self.bias_threshold_eo,
                'fpr_gap': self.bias_threshold_fpr,
                'sp_gap': self.bias_threshold_sp
            }
        }
    
    def apply_server_momentum(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply server-side momentum to model updates."""
        if not self.enable_server_momentum:
            return update
        
        if self.server_momentum_buffer is None:
            self.server_momentum_buffer = update
            return update
        
        # Apply momentum: new_update = θ * prev_update + (1-θ) * current_update
        momentum_update = {}
        for key in update.keys():
            momentum_update[key] = (
                self.theta_server * self.server_momentum_buffer[key] +
                (1 - self.theta_server) * update[key]
            )
        
        self.server_momentum_buffer = momentum_update
        return momentum_update
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'round': self.round_num,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'current_lambda': self.lambda_fair,
            'current_tau': self.tau,
            'global_metrics': {
                k: v[-1] if v else None 
                for k, v in self.global_metrics_history.items()
            },
            'client_stats': {
                cid: {
                    'participation': hist['participation_count'],
                    'avg_weight': np.mean(hist['weights']) if hist['weights'] else 0,
                    'improvement_trend': hist['improvement_trend'],
                    'stability': hist['stability_score']
                }
                for cid, hist in self.client_history.items()
            }
        }
