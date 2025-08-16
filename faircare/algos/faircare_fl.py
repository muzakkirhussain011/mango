"""FairCare-FL++: Next-generation fairness-aware federated learning algorithm."""
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@register_aggregator("faircare_fl")
class FairCareAggregator(BaseAggregator):
    """
    FairCare-FL++: Advanced fairness-aware aggregation with adaptive bias mitigation.
    
    Key features:
    1. Dual-level fairness optimization (client and server)
    2. Adaptive bias detection and mitigation modes
    3. Multi-metric fairness scoring with momentum
    4. Dynamic parameter adjustment based on bias levels
    """
    
    def __init__(
        self,
        n_clients: int,
        # Fairness metric weights
        alpha: float = 1.0,          # EO gap weight
        beta: float = 0.5,           # FPR gap weight  
        gamma: float = 0.5,          # SP gap weight
        delta: float = 0.2,          # Accuracy weight in score
        delta_init: float = 0.2,     # Initial delta value
        delta_min: float = 0.01,     # Minimum delta in bias mode
        
        # Temperature parameters
        tau: float = 1.0,            # Temperature for softmax
        tau_init: float = 1.0,       # Initial temperature
        tau_min: float = 0.1,        # Minimum temperature
        tau_anneal_rate: float = 0.95,  # Annealing factor per round
        
        # Fairness penalty parameters
        lambda_fair_init: float = 0.1,  # Initial fairness penalty
        lambda_fair_min: float = 0.01,  # Minimum lambda
        lambda_fair_max: float = 2.0,   # Maximum lambda
        lambda_adapt_rate: float = 1.2, # Lambda increase factor
        
        # Momentum parameters
        mu_client: float = 0.9,         # Client score momentum
        theta_server: float = 0.8,      # Server update momentum
        
        # Bias detection thresholds
        thr_eo: float = 0.15,           # EO gap threshold
        thr_fpr: float = 0.15,          # FPR gap threshold
        thr_sp: float = 0.10,           # SP gap threshold
        
        # Fairness loss weights (for client training)
        w_eo: float = 1.0,              # EO loss weight
        w_fpr: float = 0.5,             # FPR loss weight
        w_sp: float = 0.5,              # SP loss weight
        
        # Weight bounds
        epsilon: float = 0.01,          # Weight floor
        weight_clip: float = 10.0,      # Max weight multiplier
        
        # Score adjustments
        improvement_bonus: float = 0.1,    # Bonus for improving clients
        variance_penalty: float = 0.1,     # Penalty for unstable clients
        participation_boost: float = 0.15,  # Boost for new/rare clients
        
        fairness_metric: str = "eo_gap",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip, 
                        fairness_metric=fairness_metric)
        
        # Store all parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.delta_init = delta_init
        self.delta_min = delta_min
        
        self.tau = tau
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        
        self.lambda_fair = lambda_fair_init
        self.lambda_fair_init = lambda_fair_init
        self.lambda_fair_min = lambda_fair_min
        self.lambda_fair_max = lambda_fair_max
        self.lambda_adapt_rate = lambda_adapt_rate
        
        self.mu_client = mu_client
        self.theta_server = theta_server
        
        self.thr_eo = thr_eo
        self.thr_fpr = thr_fpr
        self.thr_sp = thr_sp
        
        self.w_eo = w_eo
        self.w_fpr = w_fpr
        self.w_sp = w_sp
        
        self.improvement_bonus = improvement_bonus
        self.variance_penalty = variance_penalty
        self.participation_boost = participation_boost
        
        # State tracking
        self.round_num = 0
        self.bias_mitigation_mode = False
        self.consecutive_bias_rounds = 0
        self.prev_global_update = None
        
        # Client history tracking
        self.client_history = {}
        for i in range(n_clients):
            self.client_history[i] = {
                'smoothed_score': 0.5,
                'fairness_scores': [],
                'accuracy_scores': [],
                'eo_gaps': [],
                'fpr_gaps': [],
                'sp_gaps': [],
                'participation_count': 0,
                'last_round': -1,
                'improvement_trend': 0.0,
                'stability_score': 1.0,
                'last_update_quality': 0.5
            }
        
        # Global metrics history
        self.global_metrics_history = {
            'avg_eo_gap': [],
            'avg_fpr_gap': [],
            'avg_sp_gap': [],
            'bias_mode': [],
            'lambda_fair': [],
            'delta': [],
            'tau': []
        }
    
    def get_fairness_config(self) -> Dict[str, Any]:
        """
        Get current fairness configuration to send to clients.
        """
        return {
            'lambda_fair': self.lambda_fair,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'w_eo': self.w_eo,
            'w_fpr': self.w_fpr,
            'w_sp': self.w_sp,
            'round': self.round_num
        }
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute FairCare-FL++ weights with adaptive bias mitigation.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Collect metrics for bias detection
        eo_gaps = []
        fpr_gaps = []
        sp_gaps = []
        
        scores = []
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            
            # Extract metrics
            eo_gap = summary.get("eo_gap", summary.get("val_EO_gap", 0.5))
            fpr_gap = summary.get("fpr_gap", summary.get("val_FPR_gap", 0.5))
            sp_gap = summary.get("sp_gap", summary.get("val_SP_gap", 0.5))
            
            eo_gaps.append(eo_gap)
            fpr_gaps.append(fpr_gap)
            sp_gaps.append(abs(sp_gap))
            
            # Get accuracy
            if "val_acc" in summary:
                accuracy = summary["val_acc"]
            elif "val_accuracy" in summary:
                accuracy = summary["val_accuracy"]
            elif "val_loss" in summary:
                accuracy = 1.0 / (1.0 + summary["val_loss"])
            else:
                accuracy = 0.5
            
            # Compute fairness components (higher is better)
            eo_score = 1.0 - min(eo_gap, 1.0) ** 0.5  # Square root for smoothing
            fpr_score = 1.0 - min(fpr_gap, 1.0) ** 0.5
            sp_score = 1.0 - min(abs(sp_gap), 1.0) ** 0.5
            
            # Weighted fairness score
            fairness_component = (
                self.alpha * eo_score + 
                self.beta * fpr_score + 
                self.gamma * sp_score
            ) / (self.alpha + self.beta + self.gamma)
            
            # Include worst-group F1 if available
            if "worst_group_F1" in summary:
                wg_f1 = summary["worst_group_F1"]
                fairness_component = 0.8 * fairness_component + 0.2 * wg_f1
            
            # Combine fairness and accuracy based on mode
            if self.bias_mitigation_mode:
                # In bias mode, heavily prioritize fairness
                combined_score = 0.95 * fairness_component + 0.05 * accuracy
            else:
                # Normal mode: balance fairness and accuracy
                combined_score = (1 - self.delta) * fairness_component + self.delta * accuracy
            
            # Apply client momentum
            prev_score = self.client_history[client_id]['smoothed_score']
            smoothed_score = self.mu_client * prev_score + (1 - self.mu_client) * combined_score
            
            # Update client history
            history = self.client_history[client_id]
            history['smoothed_score'] = smoothed_score
            history['fairness_scores'].append(fairness_component)
            history['accuracy_scores'].append(accuracy)
            history['eo_gaps'].append(eo_gap)
            history['fpr_gaps'].append(fpr_gap)
            history['sp_gaps'].append(sp_gap)
            history['participation_count'] += 1
            history['last_round'] = self.round_num
            
            # Compute performance adjustments
            final_score = smoothed_score
            
            # Improvement trend bonus
            if len(history['fairness_scores']) >= 3:
                recent = history['fairness_scores'][-3:]
                if recent[-1] > recent[0]:  # Improving
                    improvement = min(recent[-1] - recent[0], 1.0)
                    history['improvement_trend'] = improvement
                    final_score *= (1 + self.improvement_bonus * improvement)
            
            # Stability assessment
            if len(history['eo_gaps']) >= 3:
                recent_gaps = history['eo_gaps'][-3:]
                variance = np.var(recent_gaps)
                stability = 1.0 / (1.0 + variance * 10)  # Convert variance to stability
                history['stability_score'] = stability
                
                if stability > 0.8:
                    final_score *= 1.1  # Bonus for stable clients
                elif stability < 0.3:
                    final_score *= (1 - self.variance_penalty)  # Penalty for unstable
            
            # Update quality assessment
            history['last_update_quality'] = fairness_component
            if fairness_component > 0.7:
                final_score *= 1.2  # Bonus for high-quality updates
            
            # Participation incentive for new/rare clients
            if history['participation_count'] <= 2:
                final_score *= (1 + self.participation_boost)
            
            # Clip score to reasonable range
            final_score = max(0.0, min(final_score, 2.0))
            scores.append(final_score)
        
        # Detect and respond to bias
        avg_eo_gap = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr_gap = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp_gap = np.mean(sp_gaps) if sp_gaps else 0
        
        bias_detected = (
            avg_eo_gap > self.thr_eo or 
            avg_fpr_gap > self.thr_fpr or 
            avg_sp_gap > self.thr_sp
        )
        
        if bias_detected:
            if not self.bias_mitigation_mode:
                # Enter bias mitigation mode
                print(f"[Round {self.round_num}] Entering bias mitigation mode - "
                      f"EO: {avg_eo_gap:.3f}, FPR: {avg_fpr_gap:.3f}, SP: {avg_sp_gap:.3f}")
                self.bias_mitigation_mode = True
                self.consecutive_bias_rounds = 1
                
                # Increase fairness penalty
                self.lambda_fair = min(
                    self.lambda_fair_max,
                    self.lambda_fair * self.lambda_adapt_rate
                )
                # Drastically reduce accuracy weight
                self.delta = max(self.delta_min, self.delta * 0.5)
                # Use sharp temperature for focused aggregation
                self.tau = self.tau_min
            else:
                # Already in bias mode
                self.consecutive_bias_rounds += 1
                
                # Further increase lambda if bias persists
                if self.consecutive_bias_rounds > 2:
                    self.lambda_fair = min(
                        self.lambda_fair_max,
                        self.lambda_fair * 1.1
                    )
        else:
            if self.bias_mitigation_mode:
                # Exit bias mitigation mode
                print(f"[Round {self.round_num}] Exiting bias mitigation mode - "
                      f"EO: {avg_eo_gap:.3f}, FPR: {avg_fpr_gap:.3f}, SP: {avg_sp_gap:.3f}")
                self.bias_mitigation_mode = False
                self.consecutive_bias_rounds = 0
                
                # Relax fairness penalty
                self.lambda_fair = max(
                    self.lambda_fair_min,
                    self.lambda_fair * 0.9
                )
                # Restore some accuracy weight
                self.delta = min(self.delta_init, self.delta * 1.5)
        
        # Temperature annealing
        if not self.bias_mitigation_mode:
            # Normal annealing
            self.tau = max(self.tau_min, self.tau * self.tau_anneal_rate)
        
        # Store global metrics
        self.global_metrics_history['avg_eo_gap'].append(avg_eo_gap)
        self.global_metrics_history['avg_fpr_gap'].append(avg_fpr_gap)
        self.global_metrics_history['avg_sp_gap'].append(avg_sp_gap)
        self.global_metrics_history['bias_mode'].append(self.bias_mitigation_mode)
        self.global_metrics_history['lambda_fair'].append(self.lambda_fair)
        self.global_metrics_history['delta'].append(self.delta)
        self.global_metrics_history['tau'].append(self.tau)
        
        # Log current state
        if self.round_num % 5 == 0 or bias_detected:
            print(f"[Round {self.round_num}] λ_fair: {self.lambda_fair:.3f}, "
                  f"δ: {self.delta:.3f}, τ: {self.tau:.3f}, "
                  f"Bias Mode: {self.bias_mitigation_mode}")
        
        # Convert scores to weights via softmax
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Normalize scores for numerical stability
        if scores.std() > 0:
            scores_norm = (scores - scores.mean()) / (scores.std() + 1e-8)
        else:
            scores_norm = scores
        
        # Apply softmax with temperature
        if self.tau > 0:
            weights = torch.exp(scores_norm / self.tau)
            weights = weights / weights.sum()
        else:
            # If tau is 0, assign all weight to best client
            weights = torch.zeros_like(scores)
            weights[torch.argmax(scores)] = 1.0
        
        # Apply weight floor and clipping
        weights = self._postprocess(weights)
        
        return weights
    
    def apply_server_momentum(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply server-side momentum to aggregated update.
        """
        if self.prev_global_update is None:
            self.prev_global_update = update
            return update
        
        # Blend current with previous update
        momentum_update = {}
        for key in update.keys():
            momentum_update[key] = (
                self.theta_server * self.prev_global_update[key] +
                (1 - self.theta_server) * update[key]
            )
        
        self.prev_global_update = momentum_update
        return momentum_update
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current algorithm statistics."""
        return {
            'round': self.round_num,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'lambda_fair': self.lambda_fair,
            'delta': self.delta,
            'tau': self.tau,
            'consecutive_bias_rounds': self.consecutive_bias_rounds,
            'global_metrics': self.global_metrics_history,
            'client_participation': {
                cid: hist['participation_count'] 
                for cid, hist in self.client_history.items()
            }
        }
