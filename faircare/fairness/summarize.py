# faircare/fairness/summarize.py
from typing import Dict
import torch

def _safe_div(a, b):
    return float(a) / float(max(1, b))

def summarize_by_group(scores, y, a, already_binary=False) -> Dict[str, float]:
    if not already_binary:
        yhat = (scores >= 0.5).long()
    else:
        yhat = scores.long()
    out = {}
    acc = (yhat == y.long()).float().mean().item()
    out["accuracy"] = acc
    if a is None:
        out.update({"EO_gap": 0.0, "SP_gap": 0.0, "FPR_gap": 0.0,
                    "g0_TPR": 0.0, "g1_TPR": 0.0, "g0_FPR": 0.0, "g1_FPR": 0.0,
                    "g0_PosRate": 0.0, "g1_PosRate": 0.0})
        return out
    metrics = {}
    for g in [0, 1]:
        m = (a == g)
        if m.any():
            yg = y[m].long()
            yh = yhat[m]
            TP = ((yh == 1) & (yg == 1)).sum().item()
            FP = ((yh == 1) & (yg == 0)).sum().item()
            FN = ((yh == 0) & (yg == 1)).sum().item()
            TN = ((yh == 0) & (yg == 0)).sum().item()
            TPR = _safe_div(TP, TP + FN)
            FPR = _safe_div(FP, FP + TN)
            PR  = _safe_div(TP, TP + FP)
            Pos = _safe_div(TP + FP, TP + FP + TN + FN)
            metrics[g] = {"TPR": TPR, "FPR": FPR, "PR": PR, "Pos": Pos}
        else:
            metrics[g] = {"TPR": 0.0, "FPR": 0.0, "PR": 0.0, "Pos": 0.0}
    out["g0_TPR"] = metrics[0]["TPR"]; out["g1_TPR"] = metrics[1]["TPR"]
    out["g0_FPR"] = metrics[0]["FPR"]; out["g1_FPR"] = metrics[1]["FPR"]
    out["g0_PosRate"] = metrics[0]["Pos"]; out["g1_PosRate"] = metrics[1]["Pos"]
    out["EO_gap"] = abs(out["g0_TPR"] - out["g1_TPR"])
    out["FPR_gap"] = abs(out["g0_FPR"] - out["g1_FPR"])
    out["SP_gap"] = abs(out["g0_PosRate"] - out["g1_PosRate"])
    return out

