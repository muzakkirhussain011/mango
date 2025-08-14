# faircare/fairness/metrics.py
from __future__ import annotations

from typing import Dict, Any, Hashable, Optional
import numpy as np


def group_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group_names: Optional[Dict[Hashable, str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute per-group confusion counts.

    Returns
    -------
    dict
        {group_key: {"TP": int, "FP": int, "FN": int, "TN": int, "support": int}}
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive = np.asarray(sensitive)

    counts: Dict[str, Dict[str, int]] = {}
    for g in np.unique(sensitive):
        mask = (sensitive == g)
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))  # Correct TN logic
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        key = group_names.get(g, f"group_{g}") if group_names else f"group_{g}"
        counts[key] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "support": int(len(yt))}
    return counts


def _report_from_counts(counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    tpr, fpr, pos_rate, acc = {}, {}, {}, {}
    total = 0
    total_correct = 0

    for g, d in counts.items():
        tp, fp, fn, tn = d["TP"], d["FP"], d["FN"], d["TN"]
        p = tp + fn
        n = fp + tn
        s = p + n

        tpr[g] = (tp / p) if p > 0 else 0.0
        fpr[g] = (fp / n) if n > 0 else 0.0
        pos_rate[g] = ((tp + fp) / s) if s > 0 else 0.0
        acc[g] = ((tp + tn) / s) if s > 0 else 0.0

        total += s
        total_correct += (tp + tn)

    eo_gap = (max(tpr.values()) - min(tpr.values())) if tpr else 0.0
    fpr_gap = (max(fpr.values()) - min(fpr.values())) if fpr else 0.0
    sp_gap = (max(pos_rate.values()) - min(pos_rate.values())) if pos_rate else 0.0
    overall_acc = (total_correct / total) if total > 0 else 0.0

    return {
        "EO_gap": eo_gap,
        "FPR_gap": fpr_gap,
        "SP_gap": sp_gap,
        "accuracy": overall_acc,
        "per_group": {
            g: {"TPR": tpr[g], "FPR": fpr[g], "pos_rate": pos_rate[g], "accuracy": acc[g]}
            for g in counts.keys()
        },
    }


def fairness_report(*args, **kwargs) -> Dict[str, Any]:
    """
    Flexible API:

    1) fairness_report(counts_dict)
    2) fairness_report(y_true, y_pred, sensitive)
    3) fairness_report(y_true=..., y_pred=..., sensitive=...)
    4) fairness_report(counts=counts_dict)

    Returns
    -------
    dict with keys: EO_gap, FPR_gap, SP_gap, accuracy, per_group
    """
    # Case 1: single positional counts dict
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return _report_from_counts(args[0])

    # Case 2: keyword counts
    if "counts" in kwargs and isinstance(kwargs["counts"], dict):
        return _report_from_counts(kwargs["counts"])

    # Case 3: arrays positional/keyword
    if len(args) >= 3:
        y_true, y_pred, sensitive = args[:3]
    else:
        y_true = kwargs.get("y_true")
        y_pred = kwargs.get("y_pred")
        sensitive = kwargs.get("sensitive")

    if y_true is None or y_pred is None or sensitive is None:
        raise TypeError(
            "fairness_report() expects either a counts dict or (y_true, y_pred, sensitive)."
        )

    counts = group_confusion_counts(y_true, y_pred, sensitive)
    return _report_from_counts(counts)
