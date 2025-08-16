"""Differentiable fairness loss functions for client-side training."""
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_group_stats(
    pred_logits: torch.Tensor,
    y_true: torch.Tensor, 
    sensitive: torch.Tensor,
    epsilon: float = 1e-8
) -> Tuple[dict, dict]:
    """
    Compute soft group statistics for fairness losses.
    
    Args:
        pred_logits: Model predictions (logits)
        y_true: True labels (0/1)
        sensitive: Sensitive attribute (group membership)
        epsilon: Small constant for numerical stability
        
    Returns:
        group_stats: Dict mapping group to stats
        group_masks: Dict mapping group to boolean masks
    """
    # Get predictions as probabilities
    pred_probs = torch.sigmoid(pred_logits)
    
    # Get unique groups
    unique_groups = torch.unique(sensitive)
    
    group_stats = {}
    group_masks = {}
    
    for g in unique_groups:
        g_val = g.item()
        mask = (sensitive == g)
        group_masks[g_val] = mask
        
        # Group-specific predictions and labels
        y_g = y_true[mask].float()
        pred_g = pred_probs[mask]
        
        # Count positives and negatives
        n_pos = (y_g == 1).float().sum() + epsilon
        n_neg = (y_g == 0).float().sum() + epsilon
        n_total = mask.float().sum() + epsilon
        
        # Soft TPR: E[pred | y=1, a=g]
        tpr = (pred_g * (y_g == 1).float()).sum() / n_pos
        
        # Soft FPR: E[pred | y=0, a=g]  
        fpr = (pred_g * (y_g == 0).float()).sum() / n_neg
        
        # Soft PPR: E[pred | a=g]
        ppr = pred_g.mean()
        
        group_stats[g_val] = {
            'tpr': tpr,
            'fpr': fpr,
            'ppr': ppr,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'n_total': n_total
        }
    
    return group_stats, group_masks


def loss_equal_opportunity(
    pred_logits: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Differentiable Equal Opportunity loss.
    Penalizes difference in TPR between groups.
    
    Args:
        pred_logits: Model predictions (logits)
        y_true: True labels
        sensitive: Sensitive attribute
        epsilon: Numerical stability constant
        
    Returns:
        EO loss (scalar tensor)
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=pred_logits.device)
    
    group_stats, _ = compute_group_stats(pred_logits, y_true, sensitive, epsilon)
    
    # Compute pairwise squared differences in TPR
    tprs = [stats['tpr'] for stats in group_stats.values()]
    
    if len(tprs) < 2:
        return torch.tensor(0.0, device=pred_logits.device)
    
    loss = torch.tensor(0.0, device=pred_logits.device)
    count = 0
    
    for i in range(len(tprs)):
        for j in range(i + 1, len(tprs)):
            diff = tprs[i] - tprs[j]
            loss = loss + diff * diff
            count += 1
    
    # Average over pairs
    if count > 0:
        loss = loss / count
        
    return loss


def loss_fpr(
    pred_logits: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Differentiable False Positive Rate parity loss.
    Penalizes difference in FPR between groups.
    
    Args:
        pred_logits: Model predictions (logits)
        y_true: True labels
        sensitive: Sensitive attribute
        epsilon: Numerical stability constant
        
    Returns:
        FPR loss (scalar tensor)
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=pred_logits.device)
    
    group_stats, _ = compute_group_stats(pred_logits, y_true, sensitive, epsilon)
    
    # Compute pairwise squared differences in FPR
    fprs = [stats['fpr'] for stats in group_stats.values()]
    
    if len(fprs) < 2:
        return torch.tensor(0.0, device=pred_logits.device)
    
    loss = torch.tensor(0.0, device=pred_logits.device)
    count = 0
    
    for i in range(len(fprs)):
        for j in range(i + 1, len(fprs)):
            diff = fprs[i] - fprs[j]
            loss = loss + diff * diff
            count += 1
    
    # Average over pairs
    if count > 0:
        loss = loss / count
        
    return loss


def loss_sp(
    pred_logits: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Differentiable Statistical Parity loss.
    Penalizes difference in positive prediction rates between groups.
    
    Args:
        pred_logits: Model predictions (logits)
        y_true: True labels (unused but kept for consistency)
        sensitive: Sensitive attribute
        epsilon: Numerical stability constant
        
    Returns:
        SP loss (scalar tensor)
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=pred_logits.device)
    
    group_stats, _ = compute_group_stats(pred_logits, y_true, sensitive, epsilon)
    
    # Compute pairwise squared differences in PPR
    pprs = [stats['ppr'] for stats in group_stats.values()]
    
    if len(pprs) < 2:
        return torch.tensor(0.0, device=pred_logits.device)
    
    loss = torch.tensor(0.0, device=pred_logits.device)
    count = 0
    
    for i in range(len(pprs)):
        for j in range(i + 1, len(pprs)):
            diff = pprs[i] - pprs[j]
            loss = loss + diff * diff
            count += 1
    
    # Average over pairs
    if count > 0:
        loss = loss / count
        
    return loss


def compute_fairness_loss(
    pred_logits: torch.Tensor,
    y_true: torch.Tensor,
    sensitive: Optional[torch.Tensor],
    w_eo: float = 1.0,
    w_fpr: float = 0.5,
    w_sp: float = 0.5,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Combined fairness loss with weighted components.
    
    Args:
        pred_logits: Model predictions (logits)
        y_true: True labels
        sensitive: Sensitive attribute
        w_eo: Weight for Equal Opportunity loss
        w_fpr: Weight for FPR parity loss
        w_sp: Weight for Statistical Parity loss
        epsilon: Numerical stability constant
        
    Returns:
        Combined fairness loss
    """
    if sensitive is None:
        return torch.tensor(0.0, device=pred_logits.device)
    
    # Ensure tensors are on same device
    y_true = y_true.to(pred_logits.device)
    sensitive = sensitive.to(pred_logits.device)
    
    # Compute individual losses
    l_eo = loss_equal_opportunity(pred_logits, y_true, sensitive, epsilon)
    l_fpr = loss_fpr(pred_logits, y_true, sensitive, epsilon)
    l_sp = loss_sp(pred_logits, y_true, sensitive, epsilon)
    
    # Weighted combination
    total_weight = w_eo + w_fpr + w_sp
    if total_weight > 0:
        loss = (w_eo * l_eo + w_fpr * l_fpr + w_sp * l_sp) / total_weight
    else:
        loss = torch.tensor(0.0, device=pred_logits.device)
    
    return loss
