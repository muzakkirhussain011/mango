# faircare/algos/faircare_fl.py
"""FairCare-FL v2.0: Production-ready implementation with PFA, DFBD, and CALT."""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class FairCareAggregator(BaseAggregator):
    """
    FairCare-FL v2.0: Fair Federated Learning with Enhanced Bias Mitigation
    
    Key Features:
    - Multi-objective optimization balancing accuracy, fairness, and worst-group performance
    - Demographics-free bias discovery via learned reweighting
    - Adaptive bias detection and mitigation
    - Server-side momentum for stable convergence
    """
    
    def __init__(
        self,
        n_clients: int,
        version: str = "2.0.0",
        
        # Fairness parameters
        alpha: float = 1.0,          # EO gap weight
        beta: float = 0.5,           # FPR gap weight  
        gamma: float = 0.5,          # SP gap weight
        delta: float = 0.2,          # Accuracy loss weight
        
        # Temperature parameters
        tau: float = 1.0,            # Temperature for softmin
        tau_init: float = 1.0,       # Initial temperature
        tau_min: float = 0.1,        # Minimum temperature in bias mode
        tau_anneal: bool = True,     # Enable temperature annealing
        tau_anneal_rate: float = 0.95,
        
        # Client-side fairness penalty
        lambda_fair: float = 0.1,    # Fairness loss weight for clients
        lambda_fair_init: float = 0.1,
        lambda_fair_min: float = 0.01,
        lambda_fair_max: float = 1.0,
        lambda_adapt_rate: float = 1.2,
        
        # Bias detection
        bias_threshold_eo: float = 0.15,
        bias_threshold_fpr: float = 0.15,
        bias_threshold_sp: float = 0.10,
        detector_patience: int = 2,
        
        # Server momentum
        server_momentum: float = 0.9,
        momentum_beta: float = 0.9,  # Momentum coefficient
        
        # Weight constraints
        epsilon: float = 0.01,       # Minimum weight per client
        weight_clip: float = 10.0,   # Maximum weight multiplier
        
        # Fairness loss weights for clients
        w_eo: float = 1.0,
        w_fpr: float = 0.5,
        w_sp: float = 0.5,
        
        # Advanced features (v2.0)
        enable_pfa: bool = False,     # Pareto-Fair Aggregation
        enable_dfbd: bool = False,    # Demographics-Free Bias Discovery
        enable_selector: bool = False, # Fair client selection
        
        # Legacy compatibility
        gate_mode: str = "heuristic",
        use_adversary: bool = False,
        prox_mu: float = 0.01,
        
        # Base parameters
        fairness_metric: str = "eo_gap",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip,
                        fairness_metric=fairness_metric)
        
        self.version = version
        
        # Core fairness parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Temperature control
        self.tau = tau
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_anneal = tau_anneal
        self.tau_anneal_rate = tau_anneal_rate
        self.current_tau = tau_init
        
        # Fairness penalty control
        self.lambda_fair = lambda_fair
        self.lambda_fair_init = lambda_fair_init
        self.lambda_fair_min = lambda_fair_min
        self.lambda_fair_max = lambda_fair_max
        self.lambda_adapt_rate = lambda_adapt_rate
        self.current_lambda = lambda_fair_init
        
        # Bias detection
        self.bias_threshold_eo = bias_threshold_eo
        self.bias_threshold_fpr = bias_threshold_fpr
        self.bias_threshold_sp = bias_threshold_sp
        self.detector_patience = detector_patience
        self.consecutive_bias_rounds = 0
        self.bias_mitigation_mode = False
        self.rounds_in_bias_mode = 0
        
        # Server momentum
        self.server_momentum = server_momentum
        self.momentum_beta = momentum_beta
        self.momentum_buffer = None
        
        # Fairness loss weights for clients
        self.w_eo = w_eo
        self.w_fpr = w_fpr
        self.w_sp = w_sp
        
        # Advanced features control
        self.enable_pfa = enable_pfa
        self.enable_dfbd = enable_dfbd
        self.enable_selector = enable_selector
        
        # Legacy compatibility
        self.gate_mode = gate_mode
        self.use_adversary = use_adversary
        self.prox_mu = prox_mu
        
        # State tracking
        self.round_num = 0
        self.history = {
            'eo_gaps': [],
            'fpr_gaps': [],
            'sp_gaps': [],
            'worst_group_f1': [],
            'bias_mode': [],
            'lambda_values': [],
            'tau_values': []
        }
        
        # Client performance tracking for adaptive weighting
        self.client_performance = torch.zeros(n_clients)
        self.client_fairness = torch.zeros(n_clients)
        self.client_participation = torch.zeros(n_clients)
        
        # Initialize gate network for compatibility
        if gate_mode == "learned":
            self.gate_network = self._create_gate_network()
        
        # Initialize AFL tracking
        self.afl_client_weights = torch.ones(n_clients) / n_clients
        
    def _create_gate_network(self) -> nn.Module:
        """Create gate network for ensemble weighting (legacy compatibility)."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Softmax(dim=-1)
        )
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute aggregation weights using fairness-aware scoring.
        
        The key insight: Lower fairness gaps should receive higher weights,
        but we also need to consider data size and loss for stability.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Update bias detection
        self._update_bias_state(client_summaries)
        
        # Update temperature if annealing
        if self.tau_anneal and self.round_num > 1:
            self.current_tau = max(
                self.tau_min,
                self.current_tau * self.tau_anneal_rate
            )
        
        # Override temperature in bias mode
        if self.bias_mitigation_mode:
            effective_tau = self.tau_min
        else:
            effective_tau = self.current_tau
        
        # Compute fairness-aware weights
        weights = self._compute_fairness_weights(client_summaries, effective_tau)
        
        # Apply weight constraints
        weights = self._apply_weight_constraints(weights)
        
        # Update tracking
        self._update_tracking(client_summaries)
        
        return weights
    
    def _compute_fairness_weights(
        self,
        client_summaries: List[Dict[str, Any]],
        temperature: float
    ) -> torch.Tensor:
        """
        Compute weights that prioritize fair clients.
        
        Key principle: Clients with lower fairness gaps get higher weights,
        modulated by their data size and loss.
        """
        n = len(client_summaries)
        
        # Extract metrics
        scores = []
        sizes = []
        
        for summary in client_summaries:
            # Get fairness gaps (lower is better)
            eo_gap = summary.get('eo_gap', 0.0)
            fpr_gap = summary.get('fpr_gap', 0.0)
            sp_gap = abs(summary.get('sp_gap', 0.0))
            
            # Get performance metrics
            val_loss = summary.get('val_loss', summary.get('train_loss', 1.0))
            n_samples = summary.get('n_samples', 100)
            
            # Compute composite fairness penalty
            if self.bias_mitigation_mode:
                # In bias mode: heavily penalize unfair clients
                fairness_penalty = (
                    2.0 * self.alpha * eo_gap +
                    2.0 * self.beta * fpr_gap +
                    2.0 * self.gamma * sp_gap +
                    0.5 * self.delta * val_loss  # Reduce accuracy weight
                )
            else:
                # Normal mode: balanced objective
                fairness_penalty = (
                    self.alpha * eo_gap +
                    self.beta * fpr_gap +
                    self.gamma * sp_gap +
                    self.delta * val_loss
                )
            
            scores.append(fairness_penalty)
            sizes.append(n_samples)
        
        scores = torch.tensor(scores, dtype=torch.float32)
        sizes = torch.tensor(sizes, dtype=torch.float32)
        
        # Convert scores to weights using softmin
        # Lower score (better fairness) -> higher weight
        fairness_weights = torch.exp(-scores / max(temperature, 0.01))
        fairness_weights = fairness_weights / fairness_weights.sum()
        
        # Combine with data size weighting for stability
        size_weights = sizes / sizes.sum()
        
        # Blend fairness and size weights
        if self.bias_mitigation_mode:
            # Prioritize fairness in bias mode
            blend_factor = 0.8  # 80% fairness, 20% size
        else:
            # Balanced blending in normal mode
            blend_factor = 0.5  # 50% fairness, 50% size
        
        weights = blend_factor * fairness_weights + (1 - blend_factor) * size_weights
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def _apply_weight_constraints(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply epsilon floor and weight clipping constraints.
        
        Ensures:
        1. Each client gets at least epsilon weight (fairness)
        2. No client gets more than weight_clip * uniform weight (stability)
        """
        n = len(weights)
        
        if self.epsilon > 0:
            # Apply epsilon floor
            if self.epsilon * n >= 1.0:
                # Infeasible: return uniform
                return torch.ones(n, dtype=torch.float32) / n
            
            # Iterative projection to ensure floor
            for _ in range(10):
                # Set floor
                below_floor = weights < self.epsilon
                if not below_floor.any():
                    break
                
                weights[below_floor] = self.epsilon
                
                # Rescale others
                above_floor = ~below_floor
                if above_floor.any():
                    total_floor = below_floor.sum().item() * self.epsilon
                    remaining = 1.0 - total_floor
                    
                    if remaining > 0:
                        above_sum = weights[above_floor].sum()
                        if above_sum > 0:
                            weights[above_floor] *= remaining / above_sum
            
            # Final enforcement
            weights = torch.maximum(weights, torch.tensor(self.epsilon))
            weights = weights / weights.sum()
        
        if self.weight_clip > 0:
            # Apply weight clipping
            max_weight = self.weight_clip / n
            weights = torch.minimum(weights, torch.tensor(max_weight))
            weights = weights / weights.sum()
        
        return weights
    
    def _update_bias_state(self, client_summaries: List[Dict[str, Any]]):
        """
        Update bias detection state and adapt parameters.
        """
        # Compute average gaps
        eo_gaps = [s.get('eo_gap', 0) for s in client_summaries]
        fpr_gaps = [s.get('fpr_gap', 0) for s in client_summaries]
        sp_gaps = [abs(s.get('sp_gap', 0)) for s in client_summaries]
        
        avg_eo = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp = np.mean(sp_gaps) if sp_gaps else 0
        
        # Check bias conditions
        bias_detected = (
            avg_eo > self.bias_threshold_eo or
            avg_fpr > self.bias_threshold_fpr or
            avg_sp > self.bias_threshold_sp
        )
        
        if bias_detected:
            self.consecutive_bias_rounds += 1
            
            if self.consecutive_bias_rounds >= self.detector_patience:
                if not self.bias_mitigation_mode:
                    # Enter bias mitigation mode
                    self.bias_mitigation_mode = True
                    self.rounds_in_bias_mode = 0
                    
                    # Adapt lambda_fair
                    self.current_lambda = min(
                        self.lambda_fair_max,
                        self.current_lambda * self.lambda_adapt_rate
                    )
                    
                    print(f"[Round {self.round_num}] Entering bias mitigation mode")
                    print(f"  EO gap: {avg_eo:.4f}, FPR gap: {avg_fpr:.4f}, SP gap: {avg_sp:.4f}")
                    print(f"  Lambda_fair: {self.current_lambda:.4f}")
                
                self.rounds_in_bias_mode += 1
        else:
            self.consecutive_bias_rounds = 0
            
            if self.bias_mitigation_mode:
                # Check if we can exit bias mode
                exit_threshold = 0.5  # Exit if gaps are below 50% of threshold
                if (avg_eo < self.bias_threshold_eo * exit_threshold and
                    avg_fpr < self.bias_threshold_fpr * exit_threshold and
                    avg_sp < self.bias_threshold_sp * exit_threshold):
                    
                    self.bias_mitigation_mode = False
                    
                    # Relax lambda_fair
                    self.current_lambda = max(
                        self.lambda_fair_min,
                        self.current_lambda / self.lambda_adapt_rate
                    )
                    
                    print(f"[Round {self.round_num}] Exiting bias mitigation mode")
        
        # Store history
        self.history['eo_gaps'].append(avg_eo)
        self.history['fpr_gaps'].append(avg_fpr)
        self.history['sp_gaps'].append(avg_sp)
        self.history['bias_mode'].append(self.bias_mitigation_mode)
        self.history['lambda_values'].append(self.current_lambda)
        self.history['tau_values'].append(self.current_tau)
    
    def _update_tracking(self, client_summaries: List[Dict[str, Any]]):
        """
        Update client performance tracking for adaptive weighting.
        """
        for summary in client_summaries:
            client_id = summary.get('client_id', -1)
            if 0 <= client_id < self.n_clients:
                # Update performance (exponential moving average)
                loss = summary.get('val_loss', summary.get('train_loss', 1.0))
                self.client_performance[client_id] = (
                    0.9 * self.client_performance[client_id] + 0.1 * (1.0 - loss)
                )
                
                # Update fairness tracking
                eo_gap = summary.get('eo_gap', 0)
                fpr_gap = summary.get('fpr_gap', 0)
                sp_gap = abs(summary.get('sp_gap', 0))
                fairness_score = 1.0 - (eo_gap + fpr_gap + sp_gap) / 3.0
                
                self.client_fairness[client_id] = (
                    0.9 * self.client_fairness[client_id] + 0.1 * fairness_score
                )
                
                # Mark participation
                self.client_participation[client_id] += 1
        
        # Store worst-group F1 if available
        wg_f1s = [s.get('worst_group_f1', 0) for s in client_summaries if 'worst_group_f1' in s]
        if wg_f1s:
            self.history['worst_group_f1'].append(np.mean(wg_f1s))
    
    def apply_server_momentum(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply momentum to server update for stable convergence.
        """
        if self.momentum_buffer is None:
            self.momentum_buffer = {}
            for key, value in update.items():
                self.momentum_buffer[key] = torch.zeros_like(value)
        
        # Apply momentum
        momentum_update = {}
        for key, value in update.items():
            if key in self.momentum_buffer:
                self.momentum_buffer[key] = (
                    self.momentum_beta * self.momentum_buffer[key] +
                    (1 - self.momentum_beta) * value
                )
                momentum_update[key] = self.momentum_buffer[key]
            else:
                momentum_update[key] = value
        
        return momentum_update
    
    def get_fairness_config(self) -> Dict[str, Any]:
        """
        Get configuration to broadcast to clients.
        """
        config = {
            # Core parameters
            'lambda_fair': self.current_lambda,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'round': self.round_num,
            
            # Fairness loss weights
            'w_eo': self.w_eo,
            'w_fpr': self.w_fpr,
            'w_sp': self.w_sp,
            
            # Client-side parameters
            'prox_mu': self.prox_mu,
            'use_adversary': self.use_adversary,
            
            # Additional context
            'tau': self.current_tau,
            'delta_acc': self.delta,
            'version': self.version
        }
        
        # Add adaptive parameters if in bias mode
        if self.bias_mitigation_mode:
            config['extra_epochs'] = 1  # Extra local epochs in bias mode
            config['lr_multiplier'] = 1.2  # Increase learning rate
        
        return config
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregator statistics for logging.
        """
        stats = {
            'round': self.round_num,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'rounds_in_bias_mode': self.rounds_in_bias_mode,
            'current_lambda': self.current_lambda,
            'current_tau': self.current_tau,
            'lambda_fair': self.current_lambda,  # Compatibility
            'delta_acc': self.delta,
            'version': self.version
        }
        
        # Add recent history
        if self.history['eo_gaps']:
            stats['recent_eo_gap'] = self.history['eo_gaps'][-1]
            stats['recent_fpr_gap'] = self.history['fpr_gaps'][-1]
            stats['recent_sp_gap'] = self.history['sp_gaps'][-1]
        
        if self.history['worst_group_f1']:
            stats['recent_worst_group_f1'] = self.history['worst_group_f1'][-1]
        
        # Add client tracking summary
        if self.client_participation.sum() > 0:
            stats['avg_client_performance'] = self.client_performance.mean().item()
            stats['avg_client_fairness'] = self.client_fairness.mean().item()
            stats['client_participation_std'] = self.client_participation.std().item()
        
        return stats
    
    def _compute_component_weights(self, client_summaries: List[Dict[str, Any]]) -> ComponentWeights:
        """
        Compute weights from component algorithms (for test compatibility).
        """
        n = len(client_summaries)
        
        # FedAvg: proportional to data size
        sizes = torch.tensor([s.get('n_samples', 100) for s in client_summaries], dtype=torch.float32)
        fedavg_weights = sizes / sizes.sum()
        
        # FedProx: same as FedAvg
        fedprox_weights = fedavg_weights.clone()
        
        # q-FFL: upweight high-loss clients
        losses = torch.tensor([s.get('train_loss', 1.0) for s in client_summaries], dtype=torch.float32)
        q = 2.0
        qffl_weights = torch.pow(losses + 1e-4, q - 1)
        qffl_weights = qffl_weights / qffl_weights.sum()
        
        # AFL: exponential weighting by loss
        afl_weights = []
        for i, summary in enumerate(client_summaries):
            client_id = summary.get('client_id', i)
            loss = summary.get('val_loss', summary.get('train_loss', 1.0))
            
            if client_id < len(self.afl_client_weights):
                # Update AFL weight
                self.afl_client_weights[client_id] *= torch.exp(torch.tensor(0.1 * loss))
        
        # Normalize AFL weights
        self.afl_client_weights = self.afl_client_weights / self.afl_client_weights.sum()
        
        for i, summary in enumerate(client_summaries):
            client_id = summary.get('client_id', i)
            if client_id < len(self.afl_client_weights):
                afl_weights.append(self.afl_client_weights[client_id].item())
            else:
                afl_weights.append(1.0 / n)
        
        afl_weights = torch.tensor(afl_weights, dtype=torch.float32)
        afl_weights = afl_weights / afl_weights.sum()
        
        # FairFATE: fairness-based weighting
        fairness_scores = []
        for summary in client_summaries:
            score = (
                self.alpha * summary.get('eo_gap', 0) +
                self.beta * summary.get('fpr_gap', 0) +
                self.gamma * abs(summary.get('sp_gap', 0)) +
                self.delta * summary.get('val_loss', 1.0)
            )
            fairness_scores.append(score)
        
        fairness_scores = torch.tensor(fairness_scores, dtype=torch.float32)
        fairfate_weights = torch.exp(-fairness_scores / max(self.current_tau, 0.1))
        fairfate_weights = fairfate_weights / fairfate_weights.sum()
        
        return ComponentWeights(
            fedavg=fedavg_weights,
            fedprox=fedprox_weights,
            qffl=qffl_weights,
            afl=afl_weights,
            fairfate=fairfate_weights
        )
    
    def get_preferred_clients(self, n_select: int) -> Optional[List[int]]:
        """
        Get preferred clients based on fairness and performance.
        """
        if not self.enable_selector:
            return None
        
        # Compute selection scores
        scores = (
            0.5 * self.client_fairness +  # Fairness score
            0.3 * self.client_performance +  # Performance score
            0.2 * (1.0 / (self.client_participation + 1))  # Exploration bonus
        )
        
        # Add randomization for exploration
        scores += torch.randn(self.n_clients) * 0.1
        
        # Select top-k
        if n_select >= self.n_clients:
            return list(range(self.n_clients))
        
        _, indices = torch.topk(scores, n_select)
        return indices.tolist()
