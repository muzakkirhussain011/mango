# faircare/fairness/mitigation.py
"""Bias mitigation policy module for FedBLE."""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MitigationAction:
    """Action to take for bias mitigation."""
    lambda_fair: float
    delta_acc: float
    tau: float
    w_eo: float
    w_fpr: float
    w_sp: float
    use_adversary: bool
    extra_epochs: int
    dro_intensity: float
    enable_threshold_optimizer: bool


class MitigationPolicy:
    """
    Mitigation policy that maps detector outputs to parameter updates.
    
    Adjusts fairness penalties, weights, and training configuration
    based on detected bias patterns.
    """
    
    def __init__(
        self,
        lambda_fair_init: float = 0.1,
        lambda_fair_min: float = 0.01,
        lambda_fair_max: float = 2.0,
        lambda_adapt_rate: float = 1.2,
        delta_acc_init: float = 0.2,
        delta_acc_min: float = 0.01,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        w_eo_base: float = 1.0,
        w_fpr_base: float = 0.5,
        w_sp_base: float = 0.5,
        adversary_threshold: int = 3,
        max_extra_epochs: int = 2,
        enable_adaptive_weights: bool = True
    ):
        """
        Initialize mitigation policy.
        
        Args:
            lambda_fair_init: Initial fairness penalty
            lambda_fair_min: Minimum lambda
            lambda_fair_max: Maximum lambda
            lambda_adapt_rate: Lambda adjustment rate
            delta_acc_init: Initial accuracy weight
            delta_acc_min: Minimum accuracy weight
            tau_init: Initial temperature
            tau_min: Minimum temperature
            w_eo_base: Base EO weight
            w_fpr_base: Base FPR weight
            w_sp_base: Base SP weight
            adversary_threshold: Rounds before enabling adversary
            max_extra_epochs: Maximum extra epochs in bias mode
            enable_adaptive_weights: Enable adaptive weight adjustment
        """
        self.lambda_fair_init = lambda_fair_init
        self.lambda_fair_min = lambda_fair_min
        self.lambda_fair_max = lambda_fair_max
        self.lambda_adapt_rate = lambda_adapt_rate
        self.delta_acc_init = delta_acc_init
        self.delta_acc_min = delta_acc_min
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.w_eo_base = w_eo_base
        self.w_fpr_base = w_fpr_base
        self.w_sp_base = w_sp_base
        self.adversary_threshold = adversary_threshold
        self.max_extra_epochs = max_extra_epochs
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # Current state
        self.current_lambda = lambda_fair_init
        self.current_delta = delta_acc_init
        self.current_tau = tau_init
        self.rounds_in_bias_mode = 0
        
        # History for adaptive adjustment
        self.mitigation_history = []
    
    def compute_action(
        self,
        bias_state: 'BiasState',
        current_params: Optional[Dict[str, float]] = None
    ) -> MitigationAction:
        """
        Compute mitigation action based on bias state.
        
        Args:
            bias_state: Current bias detection state
            current_params: Current parameter values
        
        Returns:
            Mitigation action to take
        """
        # Use current params if provided
        if current_params:
            self.current_lambda = current_params.get('lambda_fair', self.current_lambda)
            self.current_delta = current_params.get('delta_acc', self.current_delta)
            self.current_tau = current_params.get('tau', self.current_tau)
        
        if bias_state.is_biased:
            # Enter or continue bias mitigation mode
            self.rounds_in_bias_mode += 1
            
            # Compute severity score
            severity = self._compute_severity(bias_state)
            
            # Adjust lambda based on severity and trend
            if bias_state.trend == "increasing":
                # Bias getting worse: aggressive increase
                new_lambda = min(
                    self.lambda_fair_max,
                    self.current_lambda * self.lambda_adapt_rate * (1 + severity)
                )
            elif bias_state.trend == "stable":
                # Moderate increase
                new_lambda = min(
                    self.lambda_fair_max,
                    self.current_lambda * self.lambda_adapt_rate
                )
            else:  # decreasing
                # Maintain current level
                new_lambda = self.current_lambda
            
            # Adjust accuracy weight (reduce in bias mode)
            new_delta = max(
                self.delta_acc_min,
                self.delta_acc_init * (1 - severity)
            )
            
            # Use sharp temperature for focused aggregation
            new_tau = self.tau_min
            
            # Adjust fairness weights based on triggered metrics
            w_eo, w_fpr, w_sp = self._adjust_weights(bias_state)
            
            # Enable adversarial debiasing for persistent bias
            use_adversary = self.rounds_in_bias_mode >= self.adversary_threshold
            
            # Extra epochs based on severity
            extra_epochs = min(
                self.max_extra_epochs,
                int(severity * self.max_extra_epochs)
            )
            
            # DRO intensity (for AFL component)
            dro_intensity = min(1.0, severity + 0.5)
            
            # Enable threshold optimizer for severe bias
            enable_threshold_optimizer = severity > 0.7
            
        else:
            # Exit or stay out of bias mitigation mode
            self.rounds_in_bias_mode = 0
            
            # Gradually relax fairness pressure
            new_lambda = max(
                self.lambda_fair_min,
                self.current_lambda * 0.9
            )
            
            # Restore accuracy weight
            new_delta = min(
                self.delta_acc_init,
                self.current_delta * 1.1
            )
            
            # Restore temperature
            new_tau = min(
                self.tau_init,
                self.current_tau * 1.1
            )
            
            # Use base weights
            w_eo = self.w_eo_base
            w_fpr = self.w_fpr_base
            w_sp = self.w_sp_base
            
            # Disable special features
            use_adversary = False
            extra_epochs = 0
            dro_intensity = 0.0
            enable_threshold_optimizer = False
        
        # Update current state
        self.current_lambda = new_lambda
        self.current_delta = new_delta
        self.current_tau = new_tau
        
        # Create action
        action = MitigationAction(
            lambda_fair=new_lambda,
            delta_acc=new_delta,
            tau=new_tau,
            w_eo=w_eo,
            w_fpr=w_fpr,
            w_sp=w_sp,
            use_adversary=use_adversary,
            extra_epochs=extra_epochs,
            dro_intensity=dro_intensity,
            enable_threshold_optimizer=enable_threshold_optimizer
        )
        
        # Log action
        self.mitigation_history.append({
            'round': len(self.mitigation_history),
            'is_biased': bias_state.is_biased,
            'severity': severity if bias_state.is_biased else 0,
            'action': action
        })
        
        return action
    
    def _compute_severity(self, bias_state: 'BiasState') -> float:
        """
        Compute bias severity score.
        
        Args:
            bias_state: Current bias state
        
        Returns:
            Severity score between 0 and 1
        """
        # Normalize gaps to [0, 1]
        eo_severity = min(1.0, bias_state.eo_gap / 0.3)
        fpr_severity = min(1.0, bias_state.fpr_gap / 0.3)
        sp_severity = min(1.0, bias_state.sp_gap / 0.2)
        
        # Worst-group F1 severity (inverted)
        wgf1_severity = max(0, 1.0 - bias_state.worst_group_f1)
        
        # Combine with weights
        severity = (
            0.3 * eo_severity +
            0.2 * fpr_severity +
            0.2 * sp_severity +
            0.3 * wgf1_severity
        )
        
        # Boost severity if multiple metrics triggered
        n_triggered = len(bias_state.triggered_metrics)
        if n_triggered > 2:
            severity = min(1.0, severity * 1.2)
        
        # Factor in confidence
        severity *= bias_state.confidence
        
        return min(1.0, severity)
    
    def _adjust_weights(self, bias_state: 'BiasState') -> Tuple[float, float, float]:
        """
        Adjust fairness loss weights based on which metrics are problematic.
        
        Args:
            bias_state: Current bias state
        
        Returns:
            Adjusted (w_eo, w_fpr, w_sp)
        """
        if not self.enable_adaptive_weights:
            return self.w_eo_base, self.w_fpr_base, self.w_sp_base
        
        # Start with base weights
        w_eo = self.w_eo_base
        w_fpr = self.w_fpr_base
        w_sp = self.w_sp_base
        
        # Boost weights for triggered metrics
        boost_factor = 1.5
        
        if 'eo_gap' in bias_state.triggered_metrics:
            w_eo *= boost_factor
        if 'fpr_gap' in bias_state.triggered_metrics:
            w_fpr *= boost_factor
        if 'sp_gap' in bias_state.triggered_metrics:
            w_sp *= boost_factor
        
        # Further adjust based on relative severity
        gaps = [bias_state.eo_gap, bias_state.fpr_gap, bias_state.sp_gap]
        max_gap = max(gaps) if max(gaps) > 0 else 1.0
        
        # Increase weight for the worst metric
        if bias_state.eo_gap == max_gap:
            w_eo *= 1.2
        if bias_state.fpr_gap == max_gap:
            w_fpr *= 1.2
        if bias_state.sp_gap == max_gap:
            w_sp *= 1.2
        
        # Normalize to maintain scale
        total = w_eo + w_fpr + w_sp
        if total > 0:
            scale = (self.w_eo_base + self.w_fpr_base + self.w_sp_base) / total
            w_eo *= scale
            w_fpr *= scale
            w_sp *= scale
        
        return w_eo, w_fpr, w_sp
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of mitigation history.
        
        Returns:
            Summary statistics
        """
        if not self.mitigation_history:
            return {}
        
        biased_rounds = sum(1 for h in self.mitigation_history if h['is_biased'])
        total_rounds = len(self.mitigation_history)
        
        severities = [h['severity'] for h in self.mitigation_history if h['is_biased']]
        
        return {
            'total_rounds': total_rounds,
            'biased_rounds': biased_rounds,
            'bias_percentage': biased_rounds / total_rounds * 100 if total_rounds > 0 else 0,
            'avg_severity': np.mean(severities) if severities else 0,
            'max_severity': max(severities) if severities else 0,
            'current_lambda': self.current_lambda,
            'current_delta': self.current_delta,
            'current_tau': self.current_tau
        }
    
    def reset(self):
        """
        Reset policy state.
        """
        self.current_lambda = self.lambda_fair_init
        self.current_delta = self.delta_acc_init
        self.current_tau = self.tau_init
        self.rounds_in_bias_mode = 0
        self.mitigation_history.clear()


from typing import Tuple
