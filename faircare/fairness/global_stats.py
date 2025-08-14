"""Global fairness statistics and tracking."""
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing import Tuple
from collections import defaultdict
import torch


class FairnessTracker:
    """Track fairness metrics across rounds."""
    
    def __init__(self):
        self.history = defaultdict(list)
        self.round_data = []
    
    def update(self, metrics: Dict, round_idx: int):
        """Update with new round metrics."""
        self.round_data.append({
            "round": round_idx,
            **metrics
        })
        
        for key, value in metrics.items():
            self.history[key].append(value)
    
    def get_worst_group_stats(self) -> Dict:
        """Get worst-group statistics."""
        if not self.round_data:
            return {}
        
        latest = self.round_data[-1]
        
        worst_stats = {}
        for metric in ["accuracy", "f1", "tpr"]:
            g0_key = f"g0_{metric}"
            g1_key = f"g1_{metric}"
            
            if g0_key in latest and g1_key in latest:
                worst_stats[f"worst_{metric}"] = min(
                    latest[g0_key],
                    latest[g1_key]
                )
        
        return worst_stats
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        
        for key, values in self.history.items():
            if values and isinstance(values[0], (int, float)):
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "final": values[-1]
                }
        
        return summary
    
    def bootstrap_ci(
        self,
        metric: str,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for a metric."""
        if metric not in self.history:
            return (0.0, 0.0)
        
        values = self.history[metric]
        n = len(values)
        
        if n == 0:
            return (0.0, 0.0)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper


class StreamingStats:
    """Streaming statistics computation."""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, value: float):
        """Update with new value using Welford's algorithm."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        if self.n == 0:
            return {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        variance = self.M2 / self.n if self.n > 1 else 0.0
        
        return {
            "n": self.n,
            "mean": self.mean,
            "std": np.sqrt(variance),
            "min": self.min_val,
            "max": self.max_val
        }
