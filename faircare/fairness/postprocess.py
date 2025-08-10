# faircare/fairness/postprocess.py
from typing import Dict
import torch

def _best_threshold_for_eo(p, y, a, grid=None):
    if grid is None:
        grid = torch.linspace(0.05, 0.95, steps=19)
    best = None
    for t0 in grid:
        for t1 in grid:
            yhat = torch.where(a == 0, (p >= t0), (p >= t1)).long()
            # compute TPR gap
            def _stats(mask):
                if mask.sum() == 0:
                    return 0.0, 0.0
                yg = y[mask]
                yh = yhat[mask]
                TP = ((yh == 1) & (yg == 1)).sum().item()
                FN = ((yh == 0) & (yg == 1)).sum().item()
                TPR = float(TP) / float(max(1, TP + FN))
                acc = (yh == yg).float().mean().item()
                return TPR, acc
            T0, A0 = _stats(a == 0)
            T1, A1 = _stats(a == 1)
            gap = abs(T0 - T1)
            acc = (A0 + A1) / 2.0
            score = gap - 0.05 * acc  # prioritize small gap, keep some utility
            if best is None or score < best[0]:
                best = (score, float(t0), float(t1))
    return {"t_g0": best[1], "t_g1": best[2]}

def equalized_odds_thresholds_from_counts(p, y, a):
    # For simplicity, we compute thresholds from full validation predictions.
    return _best_threshold_for_eo(p, y, a)

def apply_group_thresholds(p, a, th: Dict[str, float]):
    t0 = th.get("t_g0", 0.5)
    t1 = th.get("t_g1", 0.5)
    return torch.where(a == 0, (p >= t0).long(), (p >= t1).long())
