"""Evaluation utilities."""
from typing import Dict, Tuple, Optional, Any, List

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
import numpy as np
from scipy import stats

from faircare.fairness.metrics import fairness_report, group_confusion_counts
from faircare.core.utils import Logger


class Evaluator:
    """Model evaluation with fairness metrics."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        prefix: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        X, y, a = test_data
        device = next(model.parameters()).device
        X = X.to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            
            # FIX: Handle shape properly
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            elif outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            y_pred = (torch.sigmoid(outputs) > 0.5).int().cpu()
        
        # Basic metrics
        accuracy = (y_pred == y).float().mean().item()
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, f1_score
        cm = confusion_matrix(y.numpy(), y_pred.numpy())
        f1 = f1_score(y.numpy(), y_pred.numpy(), average='macro')
        
        metrics = {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_macro_f1": f1,
        }
        
        # Fairness metrics if sensitive attribute available
        if a is not None:
            fair_report = fairness_report(y_pred, y, a)
            for key, value in fair_report.items():
                metrics[f"{prefix}_{key}"] = value
            
            # Group-specific metrics
            group_stats = group_confusion_counts(y_pred, y, a)
            metrics[f"{prefix}_group_stats"] = group_stats
        
        return metrics
    
    def compare_algorithms(
        self,
        results_dict: Dict[str, List[Dict[str, float]]],
        metric: str = "worst_group_F1",
        test: str = "permutation"
    ) -> Dict[str, Any]:
        """Compare algorithms with statistical tests."""
        comparisons = {}
        
        baseline_algo = "fedavg"
        if baseline_algo not in results_dict:
            baseline_algo = list(results_dict.keys())[0]
        
        # FIX: Handle different metric key formats
        def get_metric_value(result, metric_name):
            """Try different key formats for the metric."""
            possible_keys = [
                metric_name,
                f"final_{metric_name}",
                f"test_{metric_name}",
                f"val_{metric_name}"
            ]
            for key in possible_keys:
                if key in result:
                    return result[key]
            # Return 0 if metric not found
            return 0.0
        
        baseline_values = [get_metric_value(r, metric) for r in results_dict[baseline_algo]]
        
        for algo_name, algo_results in results_dict.items():
            if algo_name == baseline_algo:
                continue
            
            algo_values = [get_metric_value(r, metric) for r in algo_results]
            
            # Skip comparison if all values are 0
            if all(v == 0 for v in algo_values) or all(v == 0 for v in baseline_values):
                comparisons[algo_name] = {
                    "mean": np.mean(algo_values) if algo_values else 0,
                    "std": np.std(algo_values) if algo_values else 0,
                    "baseline_mean": np.mean(baseline_values) if baseline_values else 0,
                    "baseline_std": np.std(baseline_values) if baseline_values else 0,
                    "p_value": 1.0,
                    "significant": False
                }
                continue
            
            # Paired test
            if test == "permutation":
                # Permutation test
                def test_statistic(x, y):
                    return np.mean(x) - np.mean(y)
                
                observed_diff = test_statistic(algo_values, baseline_values)
                
                # Generate permutations
                n_permutations = 10000
                permuted_diffs = []
                combined = algo_values + baseline_values
                n = len(algo_values)
                
                for _ in range(n_permutations):
                    np.random.shuffle(combined)
                    perm_algo = combined[:n]
                    perm_baseline = combined[n:]
                    permuted_diffs.append(test_statistic(perm_algo, perm_baseline))
                
                # Compute p-value
                p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
                
            elif test == "wilcoxon":
                # Wilcoxon signed-rank test
                try:
                    _, p_value = stats.wilcoxon(algo_values, baseline_values)
                except:
                    p_value = 1.0
            
            else:
                # Paired t-test
                try:
                    _, p_value = stats.ttest_rel(algo_values, baseline_values)
                except:
                    p_value = 1.0
            
            comparisons[algo_name] = {
                "mean": np.mean(algo_values),
                "std": np.std(algo_values),
                "baseline_mean": np.mean(baseline_values),
                "baseline_std": np.std(baseline_values),
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
        return comparisons
    
    def bootstrap_confidence_interval(
        self,
        values: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        bootstrap_means = []
        n = len(values)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper
