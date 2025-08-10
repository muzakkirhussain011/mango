# faircare/fairness/metrics.py
from typing import Dict
import torch

def compute_group_counts_from_batch(logits, y, a, thresh_bins=20) -> Dict[str, int]:
    """
    Build confusion counts at default threshold 0.5 (for training signals)
    and aggregate basic TPR/FPR stats. We keep it simple and unbiased.
    """
    out = {}
    if a is None:
        return out
    p = torch.sigmoid(logits)
    yhat = (p >= 0.5).long()
    for g in [0, 1]:
        m = (a == g)
        if m.any():
            yg = y[m].long()
            yh = yhat[m]
            TP = int(((yh == 1) & (yg == 1)).sum().item())
            FP = int(((yh == 1) & (yg == 0)).sum().item())
            FN = int(((yh == 0) & (yg == 1)).sum().item())
            TN = int(((yh == 0) & (yg == 0)).sum().item())
            out[f"TP_g{g}"] = out.get(f"TP_g{g}", 0) + TP
            out[f"FP_g{g}"] = out.get(f"FP_g{g}", 0) + FP
            out[f"FN_g{g}"] = out.get(f"FN_g{g}", 0) + FN
            out[f"TN_g{g}"] = out.get(f"TN_g{g}", 0) + TN
    return out
