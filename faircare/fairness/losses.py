# faircare/fairness/losses.py
"""Differentiable fairness loss functions for FedBLE."""
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_group_stats_soft(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Compute soft group statistics for differentiable fairness losses.
    
    Args:
        logits: Model outputs (logits)
        labels: True labels (0/1)
        sensitive: Sensitive attribute values
        epsilon: Small constant for numerical stability
    
    Returns:
        Dictionary mapping group to soft statistics
    """
    probs = torch.sigmoid(logits.squeeze())
    unique_groups = torch.unique(sensitive)
    
    group_stats = {}
    for g in unique_groups:
        g_val = g.item()
        mask = (sensitive == g)
        
        if mask.sum() == 0:
            continue
        
        group_labels = labels[mask].float()
        group_probs = probs[mask]
        
        # Soft true positive rate: E[pred | y=1, a=g]
        pos_mask = (group_labels == 1)
        if pos_mask.sum() > 0:
            tpr = group_probs[pos_mask].mean()
        else:
            tpr = torch.tensor(0.0, device=logits.device)
        
        # Soft false positive rate: E[pred | y=0, a=g]
        neg_mask = (group_labels == 0)
        if neg_mask.sum() > 0:
            fpr = group_probs[neg_mask].mean()
        else:
            fpr = torch.tensor(0.0, device=logits.device)
        
        # Positive prediction rate: E[pred | a=g]
        ppr = group_probs.mean()
        
        group_stats[g_val] = {
            'tpr': tpr,
            'fpr': fpr,
            'ppr': ppr,
            'n_pos': pos_mask.sum(),
            'n_neg': neg_mask.sum(),
            'n_total': mask.sum()
        }
    
    return group_stats


def equal_opportunity_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Differentiable Equal Opportunity loss (TPR parity).
    
    Penalizes squared pairwise differences in TPR between groups.
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    group_stats = compute_group_stats_soft(logits, labels, sensitive, epsilon)
    
    if len(group_stats) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Compute pairwise squared differences
    tprs = [stats['tpr'] for stats in group_stats.values()]
    
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    count = 0
    
    for i in range(len(tprs)):
        for j in range(i + 1, len(tprs)):
            diff = tprs[i] - tprs[j]
            loss = loss + diff * diff
            count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


def false_positive_rate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Differentiable False Positive Rate parity loss.
    
    Penalizes squared pairwise differences in FPR between groups.
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    group_stats = compute_group_stats_soft(logits, labels, sensitive, epsilon)
    
    if len(group_stats) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    fprs = [stats['fpr'] for stats in group_stats.values()]
    
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    count = 0
    
    for i in range(len(fprs)):
        for j in range(i + 1, len(fprs)):
            diff = fprs[i] - fprs[j]
            loss = loss + diff * diff
            count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


def statistical_parity_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Differentiable Statistical Parity loss (PPR parity).
    
    Penalizes squared pairwise differences in positive prediction rate.
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    group_stats = compute_group_stats_soft(logits, labels, sensitive, epsilon)
    
    if len(group_stats) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    pprs = [stats['ppr'] for stats in group_stats.values()]
    
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    count = 0
    
    for i in range(len(pprs)):
        for j in range(i + 1, len(pprs)):
            diff = pprs[i] - pprs[j]
            loss = loss + diff * diff
            count += 1
    
    if count > 0:
        loss = loss / count
    
    return loss


def max_gap_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Differentiable max gap loss.
    
    Uses smooth maximum approximation to penalize the largest gap.
    """
    if sensitive is None or len(torch.unique(sensitive)) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    group_stats = compute_group_stats_soft(logits, labels, sensitive, epsilon)
    
    if len(group_stats) < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Collect all pairwise differences
    tprs = [stats['tpr'] for stats in group_stats.values()]
    fprs = [stats['fpr'] for stats in group_stats.values()]
    pprs = [stats['ppr'] for stats in group_stats.values()]
    
    gaps = []
    
    for i in range(len(tprs)):
        for j in range(i + 1, len(tprs)):
            gaps.append(torch.abs(tprs[i] - tprs[j]))
            gaps.append(torch.abs(fprs[i] - fprs[j]))
            gaps.append(torch.abs(pprs[i] - pprs[j]))
    
    if not gaps:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Smooth maximum using LogSumExp
    gaps_tensor = torch.stack(gaps)
    temperature = 10.0  # Higher = closer to true max
    max_gap = torch.logsumexp(gaps_tensor * temperature, dim=0) / temperature
    
    return max_gap


def combined_fairness_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: Optional[torch.Tensor],
    w_eo: float = 1.0,
    w_fpr: float = 0.5,
    w_sp: float = 0.5,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Combined fairness loss with weighted components.
    
    Args:
        logits: Model outputs (logits)
        labels: True labels
        sensitive: Sensitive attribute
        w_eo: Weight for Equal Opportunity loss
        w_fpr: Weight for FPR parity loss
        w_sp: Weight for Statistical Parity loss
        epsilon: Numerical stability constant
    
    Returns:
        Combined fairness loss
    """
    if sensitive is None:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Ensure same device
    labels = labels.to(logits.device)
    sensitive = sensitive.to(logits.device)
    
    # Compute individual losses
    l_eo = equal_opportunity_loss(logits, labels, sensitive, epsilon)
    l_fpr = false_positive_rate_loss(logits, labels, sensitive, epsilon)
    l_sp = statistical_parity_loss(logits, labels, sensitive, epsilon)
    
    # Weighted combination
    total_weight = w_eo + w_fpr + w_sp
    if total_weight > 0:
        loss = (w_eo * l_eo + w_fpr * l_fpr + w_sp * l_sp) / total_weight
    else:
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    return loss


class AdversaryNetwork(nn.Module):
    """
    Adversary network for predicting sensitive attributes.
    
    Used for adversarial debiasing in FedBLE.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        n_sensitive_classes: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize adversary network.
        
        Args:
            input_dim: Input dimension (from model outputs or features)
            hidden_dims: Hidden layer dimensions
            n_sensitive_classes: Number of sensitive attribute classes
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 16]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_sensitive_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (detached from main model)
        
        Returns:
            Logits for sensitive attribute prediction
        """
        return self.network(x)


class GradientReversal(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.
    
    Multiplies gradients by -alpha during backpropagation.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    """
    Apply gradient reversal.
    
    Args:
        x: Input tensor
        alpha: Reversal strength
    
    Returns:
        Tensor with reversed gradients during backprop
    """
    return GradientReversal.apply(x, alpha)
