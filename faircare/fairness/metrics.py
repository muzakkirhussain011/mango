"""Fairness metrics and differentiable loss functions."""
from __future__ import annotations
from typing import Dict, Any, Hashable, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────
def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_bin01(x: np.ndarray) -> np.ndarray:
    """Convert predictions/labels to {0,1}."""
    arr = _to_np(x)
    if arr is None:
        return arr
    arr = np.asarray(arr)
    if arr.dtype.kind in ("f", "c"):  # float/complex -> threshold
        return (arr >= 0.5).astype(int).ravel()
    return arr.astype(int).ravel()


# ── Core metrics ──────────────────────────────────────────────────────────────
def group_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group_names: Optional[Dict[Hashable, str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Per-group confusion counts."""
    yt = _to_bin01(y_true)
    yp = _to_bin01(y_pred)
    s = _to_np(sensitive).ravel() if sensitive is not None else None

    if s is None:
        masks = [("group_0", np.ones_like(yt, dtype=bool))]
    else:
        # Build order of values
        uniq_vals: List[Any] = []
        for v in s:  # order-of-appearance unique list
            if v not in uniq_vals:
                uniq_vals.append(v)

        has_zero = any(v == 0 for v in uniq_vals)
        has_one = any(v == 1 for v in uniq_vals)
        if has_zero and has_one and len(uniq_vals) == 2:
            ordered = [1, 0]  # positive group first -> "group_0"
        else:
            ordered = uniq_vals

        masks = []
        for idx, val in enumerate(ordered):
            name = group_names.get(val, f"group_{idx}") if group_names else f"group_{idx}"
            masks.append((name, s == val))

    out: Dict[str, Dict[str, int]] = {}
    for name, m in masks:
        yt_g, yp_g = yt[m], yp[m]
        # Standard confusion-matrix cells (TP, FP, FN, TN)
        tp = int(np.sum((yt_g == 1) & (yp_g == 1)))
        fp = int(np.sum((yt_g == 0) & (yp_g == 1)))
        fn = int(np.sum((yt_g == 1) & (yp_g == 0)))
        tn = int(np.sum((yt_g == 0) & (yp_g == 0)))
        out[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": int(m.sum())}
    return out


def _rates_from_counts(c: Dict[str, int]) -> Tuple[float, float, float, float, float]:
    """Return (TPR, FPR, PPR, Precision, Recall) for a group's counts."""
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    pos = tp + fn
    neg = fp + tn
    n = pos + neg
    tpr = tp / pos if pos > 0 else 0.0
    fpr = fp / neg if neg > 0 else 0.0
    ppr = (tp + fp) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tpr
    return tpr, fpr, ppr, prec, rec


def _macro_f1_from_counts(counts: Dict[str, Dict[str, int]]) -> float:
    """Macro-F1 across groups (unweighted mean of group F1 scores)."""
    f1s = []
    for c in counts.values():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _worst_group_f1_from_counts(counts: Dict[str, Dict[str, int]]) -> float:
    """Minimum F1 across groups."""
    vals = []
    for c in counts.values():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        vals.append(f1)
    return float(min(vals)) if vals else 0.0


def fairness_report(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """Build a fairness report from arrays or counts."""
    # Case 1: counts provided
    if len(args) == 1 and isinstance(args[0], dict):
        d = args[0]
        if any(k.startswith("group_") for k in d.keys()):
            counts = d
        else:
            counts = {}
            g_idxs = sorted({k.split("_", 1)[0] for k in d.keys() if k.startswith("g") and "_" in k})
            for i, gk in enumerate(g_idxs):
                counts[f"group_{i}"] = {
                    "TP": int(d.get(f"{gk}_tp", 0)),
                    "FP": int(d.get(f"{gk}_fp", 0)),
                    "FN": int(d.get(f"{gk}_fn", 0)),
                    "TN": int(d.get(f"{gk}_tn", 0)),
                    "N": int(d.get(f"{gk}_n", 0)),
                }
        total_tp = sum(c["TP"] for c in counts.values())
        total_tn = sum(c["TN"] for c in counts.values())
        total_n = sum(c["N"] for c in counts.values())
        accuracy = (total_tp + total_tn) / total_n if total_n > 0 else 0.0
    else:
        # Case 2: arrays
        if len(args) >= 3:
            a0, a1, sensitive = args[:3]
        else:
            a0 = kwargs.get("y_pred")
            a1 = kwargs.get("y_true")
            sensitive = kwargs.get("sensitive")
        y_pred = _to_bin01(a0)
        y_true = _to_bin01(a1)
        accuracy = float(np.mean(y_pred == y_true))
        if sensitive is None:
            return {
                "accuracy": accuracy,
                "EO_gap": 0.0,
                "FPR_gap": 0.0,
                "SP_gap": 0.0,
                "max_group_gap": 0.0,
                "macro_F1": 0.0,
                "worst_group_F1": 0.0,
            }
        counts = group_confusion_counts(y_true, y_pred, _to_np(sensitive).ravel())

    # Per-group rates and gaps
    groups = sorted(counts.keys(), key=lambda k: int(k.split("_")[1]) if "_" in k else 0)
    rates = [_rates_from_counts(counts[g]) for g in groups]
    tprs = [r[0] for r in rates]
    fprs = [r[1] for r in rates]
    pprs = [r[2] for r in rates]

    eo_gap = abs(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    fpr_gap = abs(max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
    sp_gap = abs(max(pprs) - min(pprs)) if len(pprs) >= 2 else 0.0
    max_group_gap = max(eo_gap, fpr_gap, sp_gap)

    report = {
        "accuracy": float(accuracy),
        "EO_gap": float(eo_gap),
        "FPR_gap": float(fpr_gap),
        "SP_gap": float(sp_gap),
        "max_group_gap": float(max_group_gap),
        "macro_F1": _macro_f1_from_counts(counts),
        "worst_group_F1": _worst_group_f1_from_counts(counts),
    }

    # Add per-group keys
    for i, g in enumerate(groups):
        TPR, FPR, PPR, PREC, REC = rates[i]
        report[f"g{i}_TPR"] = float(TPR)
        report[f"g{i}_FPR"] = float(FPR)
        report[f"g{i}_PPR"] = float(PPR)
        report[f"g{i}_Precision"] = float(PREC)
        report[f"g{i}_Recall"] = float(REC)

    # Raw counts for downstream consumers
    report["group_stats"] = counts
    return report


# ── Differentiable Fairness Loss Functions ────────────────────────────────────
def compute_fairness_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    loss_type: str = 'eo',
    epsilon: float = 1e-7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute differentiable fairness loss for local training.
    
    Args:
        outputs: Model outputs (logits)
        labels: True labels
        sensitive: Sensitive attribute values
        loss_type: Type of fairness loss ('eo', 'sp', 'eo_sp_combined')
        epsilon: Small value for numerical stability
        device: Device for computation
    
    Returns:
        Fairness loss value
    """
    # Convert to probabilities
    probs = torch.sigmoid(outputs)
    
    # Get unique sensitive values
    unique_groups = torch.unique(sensitive)
    
    if len(unique_groups) < 2:
        # No fairness loss if only one group
        return torch.tensor(0.0, device=device)
    
    if loss_type == 'eo':
        return equal_opportunity_loss(probs, labels, sensitive, epsilon, device)
    elif loss_type == 'sp':
        return statistical_parity_loss(probs, labels, sensitive, epsilon, device)
    elif loss_type == 'eo_sp_combined':
        eo_loss = equal_opportunity_loss(probs, labels, sensitive, epsilon, device)
        sp_loss = statistical_parity_loss(probs, labels, sensitive, epsilon, device)
        return 0.6 * eo_loss + 0.4 * sp_loss
    elif loss_type == 'fpr':
        return false_positive_rate_loss(probs, labels, sensitive, epsilon, device)
    elif loss_type == 'comprehensive':
        # Comprehensive fairness loss combining multiple metrics
        eo_loss = equal_opportunity_loss(probs, labels, sensitive, epsilon, device)
        sp_loss = statistical_parity_loss(probs, labels, sensitive, epsilon, device)
        fpr_loss = false_positive_rate_loss(probs, labels, sensitive, epsilon, device)
        return 0.5 * eo_loss + 0.3 * sp_loss + 0.2 * fpr_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def equal_opportunity_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute Equal Opportunity loss (difference in TPR between groups).
    """
    unique_groups = torch.unique(sensitive)
    tprs = []
    
    for group in unique_groups:
        group_mask = (sensitive == group)
        group_positive_mask = group_mask & (labels == 1)
        
        if group_positive_mask.sum() > 0:
            # TPR for this group (soft version using probabilities)
            group_tpr = (probs[group_positive_mask]).mean()
            tprs.append(group_tpr)
    
    if len(tprs) < 2:
        return torch.tensor(0.0, device=device)
    
    # Compute pairwise differences and return squared sum
    loss = torch.tensor(0.0, device=device)
    for i in range(len(tprs)):
        for j in range(i + 1, len(tprs)):
            loss += (tprs[i] - tprs[j]) ** 2
    
    return loss


def statistical_parity_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute Statistical Parity loss (difference in positive prediction rates).
    """
    unique_groups = torch.unique(sensitive)
    pprs = []
    
    for group in unique_groups:
        group_mask = (sensitive == group)
        
        if group_mask.sum() > 0:
            # Positive prediction rate for this group (soft version)
            group_ppr = probs[group_mask].mean()
            pprs.append(group_ppr)
    
    if len(pprs) < 2:
        return torch.tensor(0.0, device=device)
    
    # Compute pairwise differences and return squared sum
    loss = torch.tensor(0.0, device=device)
    for i in range(len(pprs)):
        for j in range(i + 1, len(pprs)):
            loss += (pprs[i] - pprs[j]) ** 2
    
    return loss


def false_positive_rate_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute False Positive Rate parity loss.
    """
    unique_groups = torch.unique(sensitive)
    fprs = []
    
    for group in unique_groups:
        group_mask = (sensitive == group)
        group_negative_mask = group_mask & (labels == 0)
        
        if group_negative_mask.sum() > 0:
            # FPR for this group (soft version)
            group_fpr = probs[group_negative_mask].mean()
            fprs.append(group_fpr)
    
    if len(fprs) < 2:
        return torch.tensor(0.0, device=device)
    
    # Compute pairwise differences and return squared sum
    loss = torch.tensor(0.0, device=device)
    for i in range(len(fprs)):
        for j in range(i + 1, len(fprs)):
            loss += (fprs[i] - fprs[j]) ** 2
    
    return loss


def equalized_odds_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute Equalized Odds loss (combination of EO and FPR parity).
    """
    eo_loss = equal_opportunity_loss(probs, labels, sensitive, epsilon, device)
    fpr_loss = false_positive_rate_loss(probs, labels, sensitive, epsilon, device)
    return eo_loss + fpr_loss


def balanced_accuracy_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    epsilon: float = 1e-7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Compute balanced accuracy loss across groups.
    """
    unique_groups = torch.unique(sensitive)
    group_accuracies = []
    
    for group in unique_groups:
        group_mask = (sensitive == group)
        
        if group_mask.sum() > 0:
            # Soft accuracy for this group
            group_correct = (probs[group_mask] * labels[group_mask] + 
                           (1 - probs[group_mask]) * (1 - labels[group_mask]))
            group_acc = group_correct.mean()
            group_accuracies.append(group_acc)
    
    if len(group_accuracies) < 2:
        return torch.tensor(0.0, device=device)
    
    # Penalize variance in accuracies
    accuracies_tensor = torch.stack(group_accuracies)
    variance = torch.var(accuracies_tensor)
    
    return variance
