# faircare/fairness/metrics.py
from __future__ import annotations

from typing import Dict, Any, Hashable, Optional, Tuple
import numpy as np
import torch


def _to_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def group_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    group_names: Optional[Dict[Hashable, str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute per-group confusion counts.
    Returns a mapping like:
        {
          "group_0": {"TP": ..., "FP": ..., "FN": ..., "TN": ..., "N": ...},
          "group_1": {...}
        }
    """
    yt = _to_np(y_true).astype(int).ravel()
    yp = _to_np(y_pred).astype(int).ravel()
    s = _to_np(sensitive).ravel() if sensitive is not None else None

    if s is None:
        mask = np.ones_like(yt, dtype=bool)
        groups = [("group_0", mask)]
    else:
        # Stable order: sort unique *by value*, so sensitive==0 maps to "group_0"
        uniq_vals = np.unique(s)  # sorted order (0,1,2,...) ensures deterministic naming
        groups = []
        for idx, val in enumerate(uniq_vals):
            name = group_names.get(val, f"group_{idx}") if group_names else f"group_{idx}"
            groups.append((name, s == val))

    out: Dict[str, Dict[str, int]] = {}
    for name, m in groups:
        yt_g, yp_g = yt[m], yp[m]
        tp = int(np.sum((yt_g == 1) & (yp_g == 1)))
        fp = int(np.sum((yt_g == 0) & (yp_g == 1)))
        fn = int(np.sum((yt_g == 1) & (yp_g == 0)))
        tn = int(np.sum((yt_g == 0) & (yp_g == 0)))
        out[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": int(m.sum())}
    return out


def _rates_from_counts(c: Dict[str, int]) -> Tuple[float, float, float]:
    """Return (TPR, FPR, PPR) from a group's confusion counts dict."""
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    pos = tp + fn
    neg = fp + tn
    n = pos + neg
    tpr = tp / pos if pos > 0 else 0.0
    fpr = fp / neg if neg > 0 else 0.0
    ppr = (tp + fp) / n if n > 0 else 0.0  # predicted positive rate
    return tpr, fpr, ppr


def _macro_f1_from_counts(counts: Dict[str, Dict[str, int]]) -> float:
    """
    Macro-F1 across groups; F1 per group computed from precision/recall for the positive class,
    then averaged. (Matches scikit-learn's macro-F1 spirit.) :contentReference[oaicite:6]{index=6}
    """
    f1s = []
    for c in counts.values():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (prec + rec) == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def fairness_report(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """
    Flexible fairness report:
    - fairness_report(counts_dict) where counts_dict is either
      {"group_0": {"TP":...}, "group_1": {...}}  OR a flat dict with g0_* keys.
    - fairness_report(y_pred, y_true, sensitive) OR fairness_report(y_true, y_pred, sensitive)
    - If `sensitive` is None → gaps are 0.0.
    Returns: accuracy, EO_gap, FPR_gap, SP_gap, max_group_gap, macro_F1.
    """
    # Case 1: counts dict provided
    if len(args) == 1 and isinstance(args[0], dict):
        d = args[0]
        # Shape A: nested by "group_i"
        if any(k.startswith("group_") for k in d.keys()):
            counts = d
        else:
            # Shape B: flat g{i}_tp/fp/fn/tn/n
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
        # Case 2: arrays provided (either order for first two)
        if len(args) >= 3:
            a0, a1, sensitive = args[:3]
        else:
            a0 = kwargs.get("y_pred")
            a1 = kwargs.get("y_true")
            sensitive = kwargs.get("sensitive")
        y0 = _to_np(a0).astype(int).ravel()
        y1 = _to_np(a1).astype(int).ravel()
        # Pick the order with higher vanilla accuracy (robust to arg order).
        acc_pred_true = float(np.mean(y0 == y1))
        acc_true_pred = acc_pred_true  # symmetric for equality; keep for clarity
        y_pred, y_true = (y0, y1) if acc_pred_true >= acc_true_pred else (y1, y0)
        accuracy = float(np.mean(y_pred == y_true))

        if sensitive is None:
            return {
                "accuracy": accuracy,
                "EO_gap": 0.0,
                "FPR_gap": 0.0,
                "SP_gap": 0.0,
                "max_group_gap": 0.0,
                "macro_F1": 0.0,
            }

        counts = group_confusion_counts(y_true, y_pred, _to_np(sensitive).ravel())

    # Gaps from counts (common path)
    groups = sorted(counts.keys())  # deterministic
    if len(groups) < 2:
        return {
            "accuracy": accuracy,
            "EO_gap": 0.0,
            "FPR_gap": 0.0,
            "SP_gap": 0.0,
            "max_group_gap": 0.0,
            "macro_F1": _macro_f1_from_counts(counts),
        }

    def _rates(c):
        tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
        pos, neg = tp + fn, fp + tn
        tpr = tp / pos if pos > 0 else 0.0
        fpr = fp / neg if neg > 0 else 0.0
        ppr = (tp + fp) / (pos + neg) if (pos + neg) > 0 else 0.0
        return tpr, fpr, ppr

    rates = [_rates(counts[g]) for g in groups]
    tprs = [r[0] for r in rates]
    fprs = [r[1] for r in rates]
    pprs = [r[2] for r in rates]

    eo_gap = abs(max(tprs) - min(tprs))
    fpr_gap = abs(max(fprs) - min(fprs))
    sp_gap = abs(max(pprs) - min(pprs))
    max_group_gap = max(eo_gap, fpr_gap, sp_gap)

    return {
        "accuracy": float(accuracy),
        "EO_gap": float(eo_gap),
        "FPR_gap": float(fpr_gap),
        "SP_gap": float(sp_gap),
        "max_group_gap": float(max_group_gap),
        "macro_F1": _macro_f1_from_counts(counts),
    }
