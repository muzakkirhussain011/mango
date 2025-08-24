# ============================================================================
# faircare/utils/metrics.py
# ============================================================================

def compute_fairness_metrics(predictions: np.ndarray, targets: np.ndarray, 
                           groups: np.ndarray) -> Dict[str, float]:
    """Compute fairness metrics.
    
    Args:
        predictions: Model predictions
        targets: True labels
        groups: Sensitive group labels
        
    Returns:
        Dictionary of fairness metrics
    """
    metrics = {}
    
    unique_groups = np.unique(groups)
    
    # Compute per-group metrics
    tpr_values = []
    fpr_values = []
    ppr_values = []
    
    for group in unique_groups:
        group_mask = groups == group
        group_preds = predictions[group_mask]
        group_targets = targets[group_mask]
        
        # True Positive Rate
        positive_mask = group_targets == 1
        if positive_mask.sum() > 0:
            tpr = (group_preds[positive_mask] == 1).mean()
        else:
            tpr = 0.0
        tpr_values.append(tpr)
        
        # False Positive Rate
        negative_mask = group_targets == 0
        if negative_mask.sum() > 0:
            fpr = (group_preds[negative_mask] == 1).mean()
        else:
            fpr = 0.0
        fpr_values.append(fpr)
        
        # Positive Prediction Rate
        ppr = (group_preds == 1).mean()
        ppr_values.append(ppr)
    
    # Compute gaps
    metrics['eo_gap'] = max(tpr_values) - min(tpr_values)
    metrics['fpr_gap'] = max(fpr_values) - min(fpr_values)
    metrics['sp_gap'] = max(ppr_values) - min(ppr_values)
    
    return metrics


def compute_worst_group_metrics(predictions: np.ndarray, targets: np.ndarray,
                               groups: np.ndarray) -> Dict[str, float]:
    """Compute worst-group performance metrics.
    
    Args:
        predictions: Model predictions
        targets: True labels
        groups: Sensitive group labels
        
    Returns:
        Dictionary of worst-group metrics
    """
    metrics = {}
    
    unique_groups = np.unique(groups)
    
    f1_scores = []
    accuracies = []
    
    for group in unique_groups:
        group_mask = groups == group
        group_preds = predictions[group_mask]
        group_targets = targets[group_mask]
        
        # Compute F1 score
        tp = ((group_preds == 1) & (group_targets == 1)).sum()
        fp = ((group_preds == 1) & (group_targets == 0)).sum()
        fn = ((group_preds == 0) & (group_targets == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
        
        # Compute accuracy
        accuracy = (group_preds == group_targets).mean()
        accuracies.append(accuracy)
    
    metrics['worst_group_f1'] = min(f1_scores) if f1_scores else 0.0
    metrics['worst_group_accuracy'] = min(accuracies) if accuracies else 0.0
    
    return metrics

