# faircare/algos/faircare_fl.py
"""FedBLE: Federated Bias-Leveraging Ensemble - Next-generation fair federated learning."""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@dataclass
class ComponentWeights:
    """Weights from each component algorithm."""
    fedavg: torch.Tensor
    fedprox: torch.Tensor
    qffl: torch.Tensor
    afl: torch.Tensor
    fairfate: torch.Tensor


@register_aggregator("faircare_fl")
class FedBLEAggregator(BaseAggregator):
    """
    FedBLE: Federated Bias-Leveraging Ensemble
    
    An ensemble-style, bias-specialized FL algorithm that leverages strengths of 
    FedAvg, FedProx, q-FFL, AFL, and FairFATE while adding novel bias detection & mitigation.
    
    Key features:
    1. Dynamic per-client, per-round ensemble routing/weighting
    2. Real-time bias detection and mitigation
    3. Adversarial debiasing support
    4. Multi-metric fairness optimization
    5. Adaptive hyperparameter tuning
    """
    
    def __init__(
        self,
        n_clients: int,
        # Ensemble configuration
        gate_mode: str = "heuristic",  # "heuristic" or "learned"
        ensemble_momentum: float = 0.7,
        
        # Component algorithm parameters
        qffl_q: float = 2.0,
        qffl_eps: float = 1e-4,
        afl_eta: float = 0.1,
        afl_smoothing: float = 0.01,
        prox_mu: float = 0.01,
        
        # Fairness metric weights
        alpha: float = 1.0,          # EO gap weight
        beta: float = 0.5,           # FPR gap weight  
        gamma: float = 0.5,          # SP gap weight
        delta_acc: float = 0.2,      # Accuracy weight in score
        delta_acc_init: float = 0.2,
        delta_acc_min: float = 0.01,
        
        # Temperature parameters
        tau: float = 1.0,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal: bool = True,
        tau_anneal_rate: float = 0.95,
        
        # Fairness penalty parameters
        lambda_fair: float = 0.1,
        lambda_fair_init: float = 0.1,
        lambda_fair_min: float = 0.01,
        lambda_fair_max: float = 2.0,
        lambda_adapt_rate: float = 1.2,
        
        # Adversarial parameters
        use_adversary: bool = False,
        adv_weight: float = 0.1,
        
        # Momentum parameters
        server_momentum: float = 0.8,
        
        # Bias detection thresholds
        bias_threshold_eo: float = 0.15,
        bias_threshold_fpr: float = 0.15,
        bias_threshold_sp: float = 0.10,
        
        # Client fairness loss weights
        w_eo: float = 1.0,
        w_fpr: float = 0.5,
        w_sp: float = 0.5,
        
        # Weight bounds
        epsilon: float = 0.01,
        weight_clip: float = 10.0,
        weight_capx: float = 5.0,
        
        # Score adjustments
        improvement_bonus: float = 0.1,
        variance_penalty: float = 0.1,
        participation_boost: float = 0.15,
        
        # Detector parameters
        detector_patience: int = 2,
        hysteresis: float = 0.02,
        
        # Extra training
        extra_epoch_bias_mode: bool = True,
        
        fairness_metric: str = "eo_gap",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip,
                        fairness_metric=fairness_metric)
        
        # Store all parameters
        self.gate_mode = gate_mode
        self.ensemble_momentum = ensemble_momentum
        
        # Component algorithm params
        self.qffl_q = qffl_q
        self.qffl_eps = qffl_eps
        self.afl_eta = afl_eta
        self.afl_smoothing = afl_smoothing
        self.prox_mu = prox_mu
        
        # Fairness params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_acc = delta_acc
        self.delta_acc_init = delta_acc_init
        self.delta_acc_min = delta_acc_min
        
        self.tau = tau
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_anneal = tau_anneal
        self.tau_anneal_rate = tau_anneal_rate
        
        self.lambda_fair = lambda_fair if lambda_fair > 0 else lambda_fair_init
        self.lambda_fair_init = lambda_fair_init
        self.lambda_fair_min = lambda_fair_min
        self.lambda_fair_max = lambda_fair_max
        self.lambda_adapt_rate = lambda_adapt_rate
        
        self.use_adversary = use_adversary
        self.adv_weight = adv_weight
        
        self.server_momentum = server_momentum
        
        self.bias_threshold_eo = bias_threshold_eo
        self.bias_threshold_fpr = bias_threshold_fpr
        self.bias_threshold_sp = bias_threshold_sp
        
        self.w_eo = w_eo
        self.w_fpr = w_fpr
        self.w_sp = w_sp
        
        self.weight_capx = weight_capx
        self.improvement_bonus = improvement_bonus
        self.variance_penalty = variance_penalty
        self.participation_boost = participation_boost
        
        self.detector_patience = detector_patience
        self.hysteresis = hysteresis
        self.extra_epoch_bias_mode = extra_epoch_bias_mode
        
        # Initialize state
        self.round_num = 0
        self.bias_mitigation_mode = False
        self.consecutive_bias_rounds = 0
        self.prev_global_update = None
        
        # Initialize gating network if using learned mode
        if self.gate_mode == "learned":
            self.gate_network = self._create_gate_network()
            self.gate_optimizer = torch.optim.Adam(self.gate_network.parameters(), lr=0.001)
        
        # Component weights history for momentum
        self.prev_component_weights = None
        
        # Client history tracking
        self.client_history = {}
        for i in range(n_clients):
            self.client_history[i] = {
                'smoothed_score': 0.5,
                'component_weights': None,  # Track which components work best for each client
                'fairness_scores': [],
                'accuracy_scores': [],
                'eo_gaps': [],
                'fpr_gaps': [],
                'sp_gaps': [],
                'participation_count': 0,
                'last_round': -1,
                'improvement_trend': 0.0,
                'stability_score': 1.0,
                'delta_norms': [],
                'grad_norms': []
            }
        
        # Global metrics history
        self.global_metrics_history = {
            'avg_eo_gap': [],
            'avg_fpr_gap': [],
            'avg_sp_gap': [],
            'worst_group_f1': [],
            'bias_mode': [],
            'lambda_fair': [],
            'delta_acc': [],
            'tau': [],
            'component_mixture': []  # Track ensemble weights over time
        }
        
        # AFL client weights for min-max optimization
        self.afl_client_weights = torch.ones(n_clients) / n_clients
        
    def _create_gate_network(self) -> nn.Module:
        """Create a small MLP for learned gating."""
        return nn.Sequential(
            nn.Linear(10, 32),  # Input: standardized client stats
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),  # Output: weights for 5 components
            nn.Softmax(dim=-1)
        )
    def _compute_component_weights(
        self,
        client_summaries: List[Dict[str, Any]]
    ) -> ComponentWeights:
        """
        Compute weights from each component algorithm.
        
        Returns weights for FedAvg, FedProx, q-FFL, AFL, and FairFATE rules.
        """
        n = len(client_summaries)
        
        # FedAvg: weight by number of samples
        fedavg_weights = []
        for summary in client_summaries:
            n_samples = summary.get("n_samples", 1)
            fedavg_weights.append(n_samples)
        fedavg_weights = torch.tensor(fedavg_weights, dtype=torch.float32)
        fedavg_weights = fedavg_weights / fedavg_weights.sum()
        
        # FedProx: similar to FedAvg but will use proximal term in client training
        fedprox_weights = fedavg_weights.clone()
        
        # q-FFL: weight by (loss)^q
        qffl_weights = []
        for summary in client_summaries:
            loss = summary.get("train_loss", summary.get("val_loss", 1.0))
            weight = (loss + self.qffl_eps) ** self.qffl_q
            qffl_weights.append(weight)
        qffl_weights = torch.tensor(qffl_weights, dtype=torch.float32)
        qffl_weights = qffl_weights / qffl_weights.sum()
        
        # AFL: exponential weighting for min-max optimization
        afl_weights = []
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            loss = summary.get("val_loss", summary.get("train_loss", 1.0))
            
            # Update AFL weights using exponential gradient
            if client_id < len(self.afl_client_weights):
                # Fix: Convert to tensor before exp
                loss_tensor = torch.tensor(self.afl_eta * loss, dtype=torch.float32)
                self.afl_client_weights[client_id] *= torch.exp(loss_tensor)
        
        # Normalize and extract weights for selected clients
        self.afl_client_weights = self.afl_client_weights / self.afl_client_weights.sum()
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            if client_id < len(self.afl_client_weights):
                afl_weights.append(self.afl_client_weights[client_id].item())
            else:
                afl_weights.append(1.0 / len(client_summaries))
        afl_weights = torch.tensor(afl_weights, dtype=torch.float32)
        afl_weights = afl_weights / afl_weights.sum()
        
        # FairFATE: fairness-score guided weights
        fairfate_weights = []
        for summary in client_summaries:
            eo_gap = summary.get("eo_gap", 0.5)
            fpr_gap = summary.get("fpr_gap", 0.5)
            sp_gap = summary.get("sp_gap", 0.5)
            val_loss = summary.get("val_loss", 1.0)
            
            # Compute fairness penalty (lower is better)
            fairness_penalty = (
                self.alpha * eo_gap +
                self.beta * fpr_gap +
                self.gamma * abs(sp_gap) +
                0.1 * val_loss
            )
            fairfate_weights.append(fairness_penalty)
        
        # Convert to weights using softmin
        fairfate_weights = torch.tensor(fairfate_weights, dtype=torch.float32)
        fairfate_weights = torch.exp(-fairfate_weights / max(self.tau, 0.1))
        fairfate_weights = fairfate_weights / fairfate_weights.sum()
        
        return ComponentWeights(
            fedavg=fedavg_weights,
            fedprox=fedprox_weights,
            qffl=qffl_weights,
            afl=afl_weights,
            fairfate=fairfate_weights
        )    
    def _compute_ensemble_weights(
        self,
        client_summaries: List[Dict[str, Any]],
        component_weights: ComponentWeights
    ) -> torch.Tensor:
        """
        Compute final ensemble weights for clients using gating mechanism.
        """
        n = len(client_summaries)
        
        if self.gate_mode == "learned":
            # Use learned gating network
            final_weights = self._learned_gating(client_summaries, component_weights)
        else:
            # Use heuristic gating based on current bias state
            final_weights = self._heuristic_gating(client_summaries, component_weights)
        
        # Apply momentum to smooth transitions
        if self.prev_component_weights is not None:
            final_weights = (
                self.ensemble_momentum * self.prev_component_weights +
                (1 - self.ensemble_momentum) * final_weights
            )
        self.prev_component_weights = final_weights.clone()
        
        return final_weights
    
    def _heuristic_gating(
        self,
        client_summaries: List[Dict[str, Any]],
        component_weights: ComponentWeights
    ) -> torch.Tensor:
        """
        Heuristic gating based on bias detection and fairness metrics.
        """
        # Compute mixture coefficients based on current state
        if self.bias_mitigation_mode:
            # In bias mode: emphasize fairness-aware algorithms
            mix_coeffs = torch.tensor([
                0.1,   # FedAvg
                0.15,  # FedProx
                0.2,   # q-FFL
                0.25,  # AFL (good for worst-case)
                0.3    # FairFATE (explicitly fairness-focused)
            ])
        else:
            # Normal mode: balanced mixture
            mix_coeffs = torch.tensor([
                0.25,  # FedAvg
                0.2,   # FedProx
                0.2,   # q-FFL
                0.15,  # AFL
                0.2    # FairFATE
            ])
        
        # Adjust based on recent performance
        avg_eo_gap = np.mean([s.get("eo_gap", 0) for s in client_summaries])
        if avg_eo_gap > self.bias_threshold_eo * 1.5:
            # Severe bias: boost AFL and FairFATE
            mix_coeffs[3] *= 1.5  # AFL
            mix_coeffs[4] *= 1.5  # FairFATE
            mix_coeffs = mix_coeffs / mix_coeffs.sum()
        
        # Compute weighted combination
        final_weights = (
            mix_coeffs[0] * component_weights.fedavg +
            mix_coeffs[1] * component_weights.fedprox +
            mix_coeffs[2] * component_weights.qffl +
            mix_coeffs[3] * component_weights.afl +
            mix_coeffs[4] * component_weights.fairfate
        )
        
        # Store mixture for logging
        self.last_mixture_coeffs = mix_coeffs
        
        return final_weights
    
    def _learned_gating(
        self,
        client_summaries: List[Dict[str, Any]],
        component_weights: ComponentWeights
    ) -> torch.Tensor:
        """
        Learned gating using small MLP to predict optimal mixture.
        """
        # Prepare input features for gate network
        features_list = []
        for summary in client_summaries:
            features = torch.tensor([
                summary.get("train_loss", 1.0),
                summary.get("val_loss", 1.0),
                summary.get("eo_gap", 0.5),
                summary.get("fpr_gap", 0.5),
                summary.get("sp_gap", 0.5),
                summary.get("worst_group_f1", 0.5),
                summary.get("n_samples", 100) / 1000.0,  # Normalize
                summary.get("delta_norm", 1.0),
                summary.get("grad_norm", 1.0),
                float(self.bias_mitigation_mode)
            ], dtype=torch.float32)
            
            # Standardize features
            features = (features - features.mean()) / (features.std() + 1e-8)
            features_list.append(features)
        
        # Get mixture coefficients from gate network
        features_batch = torch.stack(features_list)
        with torch.no_grad():
            mix_coeffs = self.gate_network(features_batch).mean(dim=0)
        
        # Compute weighted combination
        final_weights = (
            mix_coeffs[0] * component_weights.fedavg +
            mix_coeffs[1] * component_weights.fedprox +
            mix_coeffs[2] * component_weights.qffl +
            mix_coeffs[3] * component_weights.afl +
            mix_coeffs[4] * component_weights.fairfate
        )
        
        # Train gate network with meta-objective (reward worst-group F1, penalize gaps)
        if self.round_num > 5:  # Start training after some rounds
            self._train_gate_network(client_summaries, mix_coeffs)
        
        self.last_mixture_coeffs = mix_coeffs
        
        return final_weights
    
    def _train_gate_network(
        self,
        client_summaries: List[Dict[str, Any]],
        predicted_mixture: torch.Tensor
    ):
        """
        Train gate network using bandit/meta-objective.
        Reward: worst-group F1, Penalty: fairness gaps
        """
        # Compute reward based on current round's performance
        worst_f1 = min([s.get("worst_group_f1", 0) for s in client_summaries])
        avg_eo_gap = np.mean([s.get("eo_gap", 0) for s in client_summaries])
        avg_fpr_gap = np.mean([s.get("fpr_gap", 0) for s in client_summaries])
        
        # Meta-objective: maximize worst F1, minimize gaps
        reward = worst_f1 - 0.5 * avg_eo_gap - 0.3 * avg_fpr_gap
        
        # Simple policy gradient update
        loss = -reward * predicted_mixture.sum()  # Negative for gradient ascent
        
        self.gate_optimizer.zero_grad()
        loss.backward()
        self.gate_optimizer.step()
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Main aggregation method using ensemble approach.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Step 1: Compute component weights from each algorithm
        component_weights = self._compute_component_weights(client_summaries)
        
        # Step 2: Detect bias and update mitigation state
        self._update_bias_detection(client_summaries)
        
        # Step 3: Adapt fairness parameters based on bias state
        self._adapt_fairness_parameters()
        
        # Step 4: Compute ensemble weights using gating
        ensemble_weights = self._compute_ensemble_weights(client_summaries, component_weights)
        
        # Step 5: Apply additional fairness-aware adjustments
        final_weights = self._apply_fairness_adjustments(ensemble_weights, client_summaries)
        
        # Step 6: Post-process weights (floor, cap, normalize)
        final_weights = self._postprocess(final_weights)
        
        # Log state
        self._log_round_state(client_summaries)
        
        return final_weights
    
    def _update_bias_detection(self, client_summaries: List[Dict[str, Any]]):
        """
        Update bias detection state based on current metrics.
        """
        # Collect metrics
        eo_gaps = [s.get("eo_gap", 0) for s in client_summaries]
        fpr_gaps = [s.get("fpr_gap", 0) for s in client_summaries]
        sp_gaps = [abs(s.get("sp_gap", 0)) for s in client_summaries]
        
        avg_eo_gap = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr_gap = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp_gap = np.mean(sp_gaps) if sp_gaps else 0
        
        # Check thresholds with hysteresis
        if self.bias_mitigation_mode:
            # Currently in bias mode: use lower threshold to exit (hysteresis)
            exit_threshold_eo = self.bias_threshold_eo - self.hysteresis
            exit_threshold_fpr = self.bias_threshold_fpr - self.hysteresis
            exit_threshold_sp = self.bias_threshold_sp - self.hysteresis
            
            if (avg_eo_gap < exit_threshold_eo and 
                avg_fpr_gap < exit_threshold_fpr and 
                avg_sp_gap < exit_threshold_sp):
                # Exit bias mode
                self.bias_mitigation_mode = False
                self.consecutive_bias_rounds = 0
                print(f"[Round {self.round_num}] Exiting bias mitigation mode")
            else:
                self.consecutive_bias_rounds += 1
        else:
            # Not in bias mode: check if we should enter
            if (avg_eo_gap > self.bias_threshold_eo or 
                avg_fpr_gap > self.bias_threshold_fpr or 
                avg_sp_gap > self.bias_threshold_sp):
                
                if self.consecutive_bias_rounds >= self.detector_patience - 1:
                    # Enter bias mode after patience
                    self.bias_mitigation_mode = True
                    print(f"[Round {self.round_num}] Entering bias mitigation mode - "
                          f"EO: {avg_eo_gap:.3f}, FPR: {avg_fpr_gap:.3f}, SP: {avg_sp_gap:.3f}")
                else:
                    self.consecutive_bias_rounds += 1
            else:
                self.consecutive_bias_rounds = 0
        
        # Update history
        self.global_metrics_history['avg_eo_gap'].append(avg_eo_gap)
        self.global_metrics_history['avg_fpr_gap'].append(avg_fpr_gap)
        self.global_metrics_history['avg_sp_gap'].append(avg_sp_gap)
        self.global_metrics_history['bias_mode'].append(self.bias_mitigation_mode)
    
    def _adapt_fairness_parameters(self):
        """
        Dynamically adapt fairness parameters based on bias state.
        """
        if self.bias_mitigation_mode:
            # Increase fairness pressure
            self.lambda_fair = min(
                self.lambda_fair_max,
                self.lambda_fair * self.lambda_adapt_rate
            )
            
            # Reduce accuracy weight
            self.delta_acc = max(
                self.delta_acc_min,
                self.delta_acc * 0.5
            )
            
            # Use sharp temperature for focused aggregation
            self.tau = self.tau_min
            
            # Enable adversarial debiasing if severe bias
            if self.consecutive_bias_rounds > 3:
                self.use_adversary = True
        else:
            # Relax fairness pressure
            self.lambda_fair = max(
                self.lambda_fair_min,
                self.lambda_fair * 0.9
            )
            
            # Restore accuracy weight
            self.delta_acc = min(
                self.delta_acc_init,
                self.delta_acc * 1.2
            )
            
            # Temperature annealing
            if self.tau_anneal:
                self.tau = max(self.tau_min, self.tau * self.tau_anneal_rate)
            
            # Disable adversarial if not needed
            if self.consecutive_bias_rounds == 0:
                self.use_adversary = False
        
        # Update history
        self.global_metrics_history['lambda_fair'].append(self.lambda_fair)
        self.global_metrics_history['delta_acc'].append(self.delta_acc)
        self.global_metrics_history['tau'].append(self.tau)
    
    def _apply_fairness_adjustments(
        self,
        weights: torch.Tensor,
        client_summaries: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Apply additional fairness-aware adjustments to weights.
        """
        adjusted_weights = weights.clone()
        
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            history = self.client_history[client_id]
            
            # Update history
            history['participation_count'] += 1
            history['last_round'] = self.round_num
            
            if "eo_gap" in summary:
                history['eo_gaps'].append(summary["eo_gap"])
            if "delta_norm" in summary:
                history['delta_norms'].append(summary["delta_norm"])
            
            # Improvement bonus
            if len(history['eo_gaps']) >= 3:
                recent = history['eo_gaps'][-3:]
                if recent[-1] < recent[0]:  # Improving
                    improvement = min(recent[0] - recent[-1], 0.5)
                    adjusted_weights[i] *= (1 + self.improvement_bonus * improvement)
            
            # Stability assessment
            if len(history['eo_gaps']) >= 3:
                variance = np.var(history['eo_gaps'][-3:])
                if variance > 0.01:  # High variance
                    adjusted_weights[i] *= (1 - self.variance_penalty)
            
            # Participation boost for new clients
            if history['participation_count'] <= 2:
                adjusted_weights[i] *= (1 + self.participation_boost)
        
        # Renormalize
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        return adjusted_weights
    
    def _log_round_state(self, client_summaries: List[Dict[str, Any]]):
        """
        Log current round state for debugging and analysis.
        """
        if self.round_num % 5 == 0 or self.bias_mitigation_mode:
            if hasattr(self, 'last_mixture_coeffs'):
                mix_str = ", ".join([f"{c:.2f}" for c in self.last_mixture_coeffs.tolist()])
                print(f"[Round {self.round_num}] Component mix: [{mix_str}], "
                      f"λ_fair: {self.lambda_fair:.3f}, "
                      f"δ_acc: {self.delta_acc:.3f}, "
                      f"τ: {self.tau:.3f}, "
                      f"Bias Mode: {self.bias_mitigation_mode}, "
                      f"Adversary: {self.use_adversary}")
        
        # Store component mixture
        if hasattr(self, 'last_mixture_coeffs'):
            self.global_metrics_history['component_mixture'].append(
                self.last_mixture_coeffs.tolist()
            )
    
    def get_fairness_config(self) -> Dict[str, Any]:
        """
        Get current fairness configuration to send to clients.
        """
        return {
            'lambda_fair': self.lambda_fair,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'use_adversary': self.use_adversary,
            'adv_weight': self.adv_weight,
            'w_eo': self.w_eo,
            'w_fpr': self.w_fpr,
            'w_sp': self.w_sp,
            'prox_mu': self.prox_mu if not self.bias_mitigation_mode else self.prox_mu * 2,
            'extra_epoch': self.extra_epoch_bias_mode and self.bias_mitigation_mode,
            'gate_mode': self.gate_mode,
            'round': self.round_num,
            'tau': self.tau,
            'delta_acc': self.delta_acc
        }
    
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
                self.server_momentum * self.prev_global_update[key] +
                (1 - self.server_momentum) * update[key]
            )
        
        self.prev_global_update = momentum_update
        return momentum_update
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current algorithm statistics.
        """
        stats = {
            'round': self.round_num,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'lambda_fair': self.lambda_fair,
            'delta_acc': self.delta_acc,
            'tau': self.tau,
            'use_adversary': self.use_adversary,
            'consecutive_bias_rounds': self.consecutive_bias_rounds,
            'gate_mode': self.gate_mode,
            'global_metrics': self.global_metrics_history
        }
        
        if hasattr(self, 'last_mixture_coeffs'):
            stats['component_mixture'] = self.last_mixture_coeffs.tolist()
        
        return stats
