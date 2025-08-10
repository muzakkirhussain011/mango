# faircare/fairness/global_stats.py
from typing import Dict, List

def _safe_div(a, b):
    return float(a) / float(max(1, b))

def compute_global_fairness_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """
    counts keys:
      TP_g0, FP_g0, FN_g0, TN_g0
      TP_g1, FP_g1, FN_g1, TN_g1
    """
    out = {}
    for g in [0, 1]:
        TP = counts.get(f"TP_g{g}", 0)
        FP = counts.get(f"FP_g{g}", 0)
        FN = counts.get(f"FN_g{g}", 0)
        TN = counts.get(f"TN_g{g}", 0)
        TPR = _safe_div(TP, TP + FN)
        FPR = _safe_div(FP, FP + TN)
        PR  = _safe_div(TP, max(1, TP + FP))
        Pos = _safe_div(TP + FP, TP + FP + TN + FN)
        out[f"g{g}_TPR"] = TPR
        out[f"g{g}_FPR"] = FPR
        out[f"g{g}_PR"]  = PR
        out[f"g{g}_PosRate"] = Pos
    out["EO_gap"] = abs(out["g0_TPR"] - out["g1_TPR"])
    out["FPR_gap"] = abs(out["g0_FPR"] - out["g1_FPR"])
    out["SP_gap"] = abs(out["g0_PosRate"] - out["g1_PosRate"])
    return out

def worst_group_focus_per_client(per_client_counts: List[Dict[str, int]], global_stats: Dict[str, float]) -> List[float]:
    """
    Returns a nonnegative focus weight per client emphasizing the disadvantaged group.
    Use TPR gap as primary signal.
    """
    g_worst = 0 if global_stats.get("g0_TPR", 0.0) < global_stats.get("g1_TPR", 0.0) else 1
    focus = []
    for c in per_client_counts:
        TP = c.get(f"TP_g{g_worst}", 0)
        FP = c.get(f"FP_g{g_worst}", 0)
        FN = c.get(f"FN_g{g_worst}", 0)
        TN = c.get(f"TN_g{g_worst}", 0)
        n = TP + FP + FN + TN
        focus.append(_safe_div(TP + FN + FP + TN, n) if n > 0 else 0.0)  # basically 1 if any coverage
        # A more nuanced score can be: proportion of worst-group samples in that client.
        # For stability, keep it binary-ish: 1 if present, 0 otherwise.
        focus[-1] = 1.0 if n > 0 else 0.0
    # normalize to [0,1]
    s = sum(focus) + 1e-12
    focus = [f / s for f in focus] if s > 0 else [0.0 for _ in focus]
    return focus
