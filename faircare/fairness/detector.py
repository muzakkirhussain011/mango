# faircare/fairness/detector.py
"""Bias detection module for FedBLE."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class BiasState:
    """Current bias detection state."""
    is_biased: bool
    triggered_metrics: List[str]
    eo_gap: float
    fpr_gap: float
    sp_gap: float
    worst_group_f1: float
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float


class BiasDetector:
    """
    Real-time bias detector with trend analysis and change-point detection.
    
    Computes rolling global and per-client EO/FPR/SP gaps, worst-group F1.
    Supports multi-group fairness and drift detection.
    """
    
    def __init__(
        self,
        thresholds: Dict[str, float] = None,
        patience: int = 2,
        hysteresis: float = 0.02,
        window_size: int = 5,
        z_score_threshold: float = 2.0,
        enable_trend_detection: bool = True,
        enable_drift_detection: bool = True
    ):
        """
        Initialize bias detector.
        
        Args:
            thresholds: Bias thresholds for each metric
            patience: Rounds to wait before triggering
            hysteresis: Threshold reduction for exiting bias mode
            window_size: Window for moving statistics
            z_score_threshold: Z-score for drift detection
            enable_trend_detection: Enable trend analysis
            enable_drift_detection: Enable gradient drift detection
        """
        self.thresholds = thresholds or {
            'eo_gap': 0.15,
            'fpr_gap': 0.15,
            'sp_gap': 0.10,
            'worst_group_f1_min': 0.6
        }
        self.patience = patience
        self.hysteresis = hysteresis
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.enable_trend_detection = enable_trend_detection
        self.enable_drift_detection = enable_drift_detection
        
        # State tracking
        self.is_biased = False
        self.consecutive_biased_rounds = 0
        self.rounds_in_bias_mode = 0
        
        # History for trend analysis
        self.eo_history = deque(maxlen=window_size)
        self.fpr_history = deque(maxlen=window_size)
        self.sp_history = deque(maxlen=window_size)
        self.wgf1_history = deque(maxlen=window_size)
        self.drift_history = deque(maxlen=window_size)
        
        # CUSUM for change-point detection
        self.cusum_eo = 0.0
        self.cusum_fpr = 0.0
        self.cusum_sp = 0.0
        self.cusum_threshold = 0.5
        
        # Per-client tracking
        self.client_bias_counts = {}
    
    def update(
        self,
        global_metrics: Dict[str, float],
        client_metrics: Optional[List[Dict[str, float]]] = None,
        gradient_norms: Optional[List[float]] = None
    ) -> BiasState:
        """
        Update detector with new metrics and return bias state.
        
        Args:
            global_metrics: Global fairness metrics
            client_metrics: Per-client metrics
            gradient_norms: Client gradient norms for drift detection
        
        Returns:
            Current bias state
        """
        # Extract metrics
        eo_gap = global_metrics.get('eo_gap', 0.0)
        fpr_gap = global_metrics.get('fpr_gap', 0.0)
        sp_gap = abs(global_metrics.get('sp_gap', 0.0))
        worst_group_f1 = global_metrics.get('worst_group_f1', 1.0)
        
        # Update history
        self.eo_history.append(eo_gap)
        self.fpr_history.append(fpr_gap)
        self.sp_history.append(sp_gap)
        self.wgf1_history.append(worst_group_f1)
        
        # Detect bias based on thresholds
        triggered_metrics = []
        
        # Use hysteresis for state transitions
        # Use hysteresis for state transitions
        if self.is_biased:
            # Currently biased: use lower thresholds to exit
            eo_threshold = self.thresholds['eo_gap'] - self.hysteresis
            fpr_threshold = self.thresholds['fpr_gap'] - self.hysteresis
            sp_threshold = self.thresholds['sp_gap'] - self.hysteresis
            wgf1_threshold = self.thresholds.get('worst_group_f1_min', 0.6) + self.hysteresis
        else:
            # Not biased: use normal thresholds
            eo_threshold = self.thresholds['eo_gap']
            fpr_threshold = self.thresholds['fpr_gap']
            sp_threshold = self.thresholds['sp_gap']
            wgf1_threshold = self.thresholds.get('worst_group_f1_min', 0.6)
        
        # Check each metric
        if eo_gap > eo_threshold:
            triggered_metrics.append('eo_gap')
        if fpr_gap > fpr_threshold:
            triggered_metrics.append('fpr_gap')
        if sp_gap > sp_threshold:
            triggered_metrics.append('sp_gap')
        if worst_group_f1 < wgf1_threshold:
            triggered_metrics.append('worst_group_f1')
        
        # Trend detection
        trend = self._detect_trend()
        
        # Drift detection
        drift_detected = False
        if gradient_norms and self.enable_drift_detection:
            drift_detected = self._detect_drift(gradient_norms)
            if drift_detected:
                triggered_metrics.append('gradient_drift')
        
        # CUSUM change-point detection
        if self._detect_change_point(eo_gap, fpr_gap, sp_gap):
            triggered_metrics.append('change_point')
        
        # Update bias state with patience
        if triggered_metrics:
            self.consecutive_biased_rounds += 1
            if self.consecutive_biased_rounds >= self.patience:
                if not self.is_biased:
                    print(f"Bias detected after {self.patience} rounds: {triggered_metrics}")
                self.is_biased = True
                self.rounds_in_bias_mode += 1
        else:
            self.consecutive_biased_rounds = 0
            if self.is_biased:
                print(f"Bias resolved, exiting bias mode after {self.rounds_in_bias_mode} rounds")
                self.is_biased = False
                self.rounds_in_bias_mode = 0
        
        # Update per-client bias tracking
        if client_metrics:
            self._update_client_tracking(client_metrics)
        
        # Compute confidence score
        confidence = self._compute_confidence()
        
        return BiasState(
            is_biased=self.is_biased,
            triggered_metrics=triggered_metrics,
            eo_gap=eo_gap,
            fpr_gap=fpr_gap,
            sp_gap=sp_gap,
            worst_group_f1=worst_group_f1,
            trend=trend,
            confidence=confidence
        )
    
    def _detect_trend(self) -> str:
        """
        Detect trend in fairness metrics.
        
        Returns:
            "increasing", "decreasing", or "stable"
        """
        if not self.enable_trend_detection or len(self.eo_history) < 3:
            return "stable"
        
        # Simple linear trend on combined metric
        combined = [
            e + f + s - w  # Higher gaps and lower wgf1 = worse
            for e, f, s, w in zip(
                self.eo_history, self.fpr_history, 
                self.sp_history, self.wgf1_history
            )
        ]
        
        # Check if increasing or decreasing
        diffs = [combined[i+1] - combined[i] for i in range(len(combined)-1)]
        avg_diff = np.mean(diffs)
        
        if avg_diff > 0.01:
            return "increasing"  # Bias getting worse
        elif avg_diff < -0.01:
            return "decreasing"  # Bias improving
        else:
            return "stable"
    
    def _detect_drift(self, gradient_norms: List[float]) -> bool:
        """
        Detect gradient drift using z-score.
        
        Args:
            gradient_norms: Current round gradient norms
        
        Returns:
            True if drift detected
        """
        if len(gradient_norms) == 0:
            return False
        
        current_mean = np.mean(gradient_norms)
        self.drift_history.append(current_mean)
        
        if len(self.drift_history) < 3:
            return False
        
        # Compute z-score
        historical_mean = np.mean(list(self.drift_history)[:-1])
        historical_std = np.std(list(self.drift_history)[:-1])
        
        if historical_std > 0:
            z_score = abs(current_mean - historical_mean) / historical_std
            return z_score > self.z_score_threshold
        
        return False
    
    def _detect_change_point(self, eo_gap: float, fpr_gap: float, sp_gap: float) -> bool:
        """
        CUSUM change-point detection.
        
        Returns:
            True if change point detected
        """
        if len(self.eo_history) < 2:
            return False
        
        # Update CUSUM statistics
        mean_eo = np.mean(self.eo_history)
        mean_fpr = np.mean(self.fpr_history)
        mean_sp = np.mean(self.sp_history)
        
        self.cusum_eo = max(0, self.cusum_eo + eo_gap - mean_eo - 0.01)
        self.cusum_fpr = max(0, self.cusum_fpr + fpr_gap - mean_fpr - 0.01)
        self.cusum_sp = max(0, self.cusum_sp + sp_gap - mean_sp - 0.01)
        
        # Check if any CUSUM exceeds threshold
        return (self.cusum_eo > self.cusum_threshold or
                self.cusum_fpr > self.cusum_threshold or
                self.cusum_sp > self.cusum_threshold)
    
    def _update_client_tracking(self, client_metrics: List[Dict[str, float]]):
        """
        Track per-client bias patterns.
        """
        for i, metrics in enumerate(client_metrics):
            client_id = metrics.get('client_id', i)
            
            if client_id not in self.client_bias_counts:
                self.client_bias_counts[client_id] = {
                    'biased_rounds': 0,
                    'total_rounds': 0,
                    'avg_eo_gap': 0.0
                }
            
            stats = self.client_bias_counts[client_id]
            stats['total_rounds'] += 1
            
            eo_gap = metrics.get('eo_gap', 0.0)
            stats['avg_eo_gap'] = (
                stats['avg_eo_gap'] * (stats['total_rounds'] - 1) + eo_gap
            ) / stats['total_rounds']
            
            if eo_gap > self.thresholds['eo_gap']:
                stats['biased_rounds'] += 1
    
    def _compute_confidence(self) -> float:
        """
        Compute confidence score for bias detection.
        
        Returns:
            Confidence score between 0 and 1
        """
        if len(self.eo_history) < 2:
            return 0.0
        
        # Base confidence on consistency of metrics
        eo_std = np.std(self.eo_history)
        fpr_std = np.std(self.fpr_history)
        sp_std = np.std(self.sp_history)
        
        # Lower variance = higher confidence
        avg_std = (eo_std + fpr_std + sp_std) / 3
        confidence = max(0, 1 - avg_std * 2)
        
        # Boost confidence if multiple metrics triggered
        if self.is_biased:
            confidence = min(1.0, confidence + 0.2 * self.consecutive_biased_rounds / self.patience)
        
        return confidence
    
    def get_client_bias_summary(self) -> Dict[str, Dict]:
        """
        Get summary of per-client bias patterns.
        
        Returns:
            Dictionary mapping client_id to bias statistics
        """
        return self.client_bias_counts.copy()
    
    def reset(self):
        """
        Reset detector state.
        """
        self.is_biased = False
        self.consecutive_biased_rounds = 0
        self.rounds_in_bias_mode = 0
        self.eo_history.clear()
        self.fpr_history.clear()
        self.sp_history.clear()
        self.wgf1_history.clear()
        self.drift_history.clear()
        self.cusum_eo = 0.0
        self.cusum_fpr = 0.0
        self.cusum_sp = 0.0
        self.client_bias_counts.clear()
