"""Fairness metrics implementation."""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import f1_score


def group_confusion_counts(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    a: Optional[torch.Tensor] = None
) -> Dict:
    """
    Compute confusion matrix counts per group.
    
    Args:
        y_pred: Predicted labels (binary)
        y_true: True labels (binary)
        a: Sensitive attribute (binary) or None
    
    Returns:
        Dictionary with TP/FP/FN/TN and N per group
    """
    y_pred = y_pred.int()
    y_true = y_true.int()
    
    if a is None:
        # Overall metrics only
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        
        return {
            "overall": {
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "N": len(y_pred)
            }
        }
    
    # Per-group metrics
    a = a.int()
    results = {}
    
    for g in [0, 1]:
        mask = (a == g)
        if mask.sum() == 0:
            results[f"group_{g}"] = {
                "TP": 0, "FP": 0, "FN": 0, "TN": 0, "N": 0
            }
            continue
        
        y_pred_g = y_pred[mask]
        y_true_g = y_true[mask]
        
        tp_g = ((y_pred_g == 1) & (y_true_g == 1)).sum().item()
        fp_g = ((y_pred_g == 1) & (y_true_g == 0)).sum().item()
        fn_g = ((y_pred_g == 0) & (y_true_g == 1)).sum().item()
        tn_g = ((y_pred_g == 0) & (y_true_g == 0)).sum().item()
        
        results[f"group_{g}"] = {
            "TP": tp_g, "FP": fp_g, "FN": fn_g, "TN": tn_g,
            "N": mask.sum().item()
        }
    
    # Overall
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    
    results["overall"] = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "N": len(y_pred)
    }
    
    return results


def fairness_report(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    a: Optional[torch.Tensor] = None
) -> Dict:
    """
    Compute comprehensive fairness metrics.
    
    Args:
        y_pred: Predicted labels (binary)
        y_true: True labels (binary)
        a: Sensitive attribute (binary) or None
    
    Returns:
        Dictionary with metrics following pseudocode spec
    """
    y_pred = y_pred.int()
    y_true = y_true.int()
    
    # Overall accuracy
    accuracy = (y_pred == y_true).float().mean().item()
    
    # F1 scores
    y_pred_np = y_pred.numpy()
    y_true_np = y_true.numpy()
    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro')
    
    result = {
        "accuracy": accuracy,
        "macro_F1": macro_f1
    }
    
    if a is None:
        # No fairness gaps without sensitive attribute
        result.update({
            "g0_TPR": np.nan, "g0_FPR": np.nan, "g0_PR": np.nan, "g0_PosRate": np.nan,
            "g1_TPR": np.nan, "g1_FPR": np.nan, "g1_PR": np.nan, "g1_PosRate": np.nan,
            "EO_gap": 0.0,
            "FPR_gap": 0.0,
            "SP_gap": 0.0,
            "max_group_gap": 0.0,
            "worst_group_F1": macro_f1
        })
        return result
    
    a = a.int()
    
    # Compute per-group metrics
    group_metrics = {}
    
    for g in [0, 1]:
        mask = (a == g)
        n_g = mask.sum().item()
        
        if n_g == 0:
            group_metrics[g] = {
                "TPR": 0.0, "FPR": 0.0, "PR": 0.0, "PosRate": 0.0, "F1": 0.0
            }
            continue
        
        y_pred_g = y_pred[mask]
        y_true_g = y_true[mask]
        
        tp_g = ((y_pred_g == 1) & (y_true_g == 1)).sum().item()
        fp_g = ((y_pred_g == 1) & (y_true_g == 0)).sum().item()
        fn_g = ((y_pred_g == 0) & (y_true_g == 1)).sum().item()
        tn_g = ((y_pred_g == 0) & (y_true_g == 0)).sum().item()
        
        # Rates with safe division
        tpr_g = tp_g / max(tp_g + fn_g, 1)
        fpr_g = fp_g / max(fp_g + tn_g, 1)
        pr_g = tp_g / max(tp_g + fp_g, 1)
        pos_rate_g = (tp_g + fp_g) / max(n_g, 1)
        
        # F1 for this group
        if len(np.unique(y_true_g.numpy())) > 1:
            f1_g = f1_score(y_true_g.numpy(), y_pred_g.numpy())
        else:
            f1_g = 0.0
        
        group_metrics[g] = {
            "TPR": tpr_g,
            "FPR": fpr_g,
            "PR": pr_g,
            "PosRate": pos_rate_g,
            "F1": f1_g
        }
    
    # Add group metrics to result
    for g in [0, 1]:
        result[f"g{g}_TPR"] = group_metrics[g]["TPR"]
        result[f"g{g}_FPR"] = group_metrics[g]["FPR"]
        result[f"g{g}_PR"] = group_metrics[g]["PR"]
        result[f"g{g}_PosRate"] = group_metrics[g]["PosRate"]
    
    # Compute gaps
    eo_gap = abs(group_metrics[0]["TPR"] - group_metrics[1]["TPR"])
    fpr_gap = abs(group_metrics[0]["FPR"] - group_metrics[1]["FPR"])
    sp_gap = abs(group_metrics[0]["PosRate"] - group_metrics[1]["PosRate"])
    max_gap = max(eo_gap, fpr_gap, sp_gap)
    
    result.update({
        "EO_gap": eo_gap,
        "FPR_gap": fpr_gap,
        "SP_gap": sp_gap,
        "max_group_gap": max_gap,
        "worst_group_F1": min(group_metrics[0]["F1"], group_metrics[1]["F1"])
    })
    
    return result


def compute_group_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    a: torch.Tensor,
    metrics: Optional[List[str]] = None
) -> Dict:
    """
    Compute specified metrics per group.
    
    Args:
        y_pred: Predictions
        y_true: True labels
        a: Sensitive attribute
        metrics: List of metrics to compute
    
    Returns:
        Dictionary of metrics per group
    """
    if metrics is None:
        metrics = ["accuracy", "tpr", "fpr", "precision", "f1"]
    
    results = {}
    
    for g in torch.unique(a):
        mask = (a == g)
        y_pred_g = y_pred[mask]
        y_true_g = y_true[mask]
        
        group_results = {}
        
        if "accuracy" in metrics:
            group_results["accuracy"] = (y_pred_g == y_true_g).float().mean().item()
        
        if "f1" in metrics and len(y_pred_g) > 0:
            group_results["f1"] = f1_score(
                y_true_g.numpy(),
                y_pred_g.numpy(),
                zero_division=0
            )
        
        # Add more metrics as needed
        tp = ((y_pred_g == 1) & (y_true_g == 1)).sum().item()
        fp = ((y_pred_g == 1) & (y_true_g == 0)).sum().item()
        fn = ((y_pred_g == 0) & (y_true_g == 1)).sum().item()
        tn = ((y_pred_g == 0) & (y_true_g == 0)).sum().item()
        
        if "tpr" in metrics:
            group_results["tpr"] = tp / max(tp + fn, 1)
        
        if "fpr" in metrics:
            group_results["fpr"] = fp / max(fp + tn, 1)
        
        if "precision" in metrics:
            group_results["precision"] = tp / max(tp + fp, 1)
        
        results[f"group_{g.item()}"] = group_results
    
    return results
