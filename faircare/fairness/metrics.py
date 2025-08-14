from __future__ import annotations

from typing import Dict, Any, Hashable, Optional, Tuple, List
import numpy as np
import torch


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
    """
    Convert predictions/labels to {0,1}.
    If float-like, apply threshold at 0.5; otherwise cast to int.
    """
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
    """
    Per-group confusion counts.

    Group naming to match tests:
      - Use the order of first appearance of values in `sensitive` to assign
        group_0, group_1, ...
      - If `group_names` is provided, use it to map values to names; otherwise
        use "group_{i}".
    """
    yt = _to_bin01(y_true)
    yp = _to_bin01(y_pred)
    s = _to_np(sensitive).ravel() if sensitive is not None else None

    if s is None:
        masks = [("group_0", np.ones_like(yt, dtype=bool))]
    else:
        # Order-of-appearance (stable) unique values
        uniq_vals: List[Any] = []
        for v in s:
            if v not in uniq_vals:
                uniq_vals.append(v)
        masks = []
        for idx, val in enumerate(uniq_vals):
            name = group_names.get(val, f"group_{idx}") if group_names else f"group_{idx}"
            masks.append((name, s == val))

    out: Dict[str, Dict[str, int]] = {}
    for name, m in masks:
        yt_g, yp_g = yt[m], yp[m]
        # Standard confusion-matrix cells: TP=(1,1), FP=(0,1), FN=(1,0), TN=(0,0)
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
    """
    Build a fairness report from either:
      • counts dict: {"group_0": {"TP":...}, "group_1": {...}} OR flat g{i}_* keys
      • arrays: (y_pred, y_true, sensitive) or (y_true, y_pred, sensitive)

    Returns:
      accuracy, EO_gap, FPR_gap, SP_gap, max_group_gap, macro_F1, worst_group_F1,
      plus per-group keys: g{i}_TPR, g{i}_FPR, g{i}_PPR, g{i}_Precision, g{i}_Recall.

    For float scores, binarization uses a 0.5 threshold before counting
    (turning scores into labels is the standard step before confusion-matrix computation). :contentReference[oaicite:3]{index=3}
    """
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

    # Add per-group keys (handy for debugging & tests)
    for i, g in enumerate(groups):
        TPR, FPR, PPR, PREC, REC = rates[i]
        report[f"g{i}_TPR"] = float(TPR)
        report[f"g{i}_FPR"] = float(FPR)
        report[f"g{i}_PPR"] = float(PPR)
        report[f"g{i}_Precision"] = float(PREC)
        report[f"g{i}_Recall"] = float(REC)

    # Raw counts for downstream consumers (not required by tests)
    report["group_stats"] = counts
    return report
