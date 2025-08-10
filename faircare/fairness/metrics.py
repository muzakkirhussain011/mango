from __future__ import annotations
from typing import Dict, Any
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def group_confusion_counts(pred: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> Dict[str, int]:
    # counts per sensitive group (binary y assumed; extends similarly for multi-class with per-class handling)
    out = {}
    for g in torch.unique(a):
        m = a == g
        py = pred[m]; yy = y[m]
        tp = ((py == 1) & (yy == 1)).sum().item()
        tn = ((py == 0) & (yy == 0)).sum().item()
        fp = ((py == 1) & (yy == 0)).sum().item()
        fn = ((py == 0) & (yy == 1)).sum().item()
        n = m.sum().item()
        out[f"g{int(g)}_tp"] = out.get(f"g{int(g)}_tp", 0) + tp
        out[f"g{int(g)}_tn"] = out.get(f"g{int(g)}_tn", 0) + tn
        out[f"g{int(g)}_fp"] = out.get(f"g{int(g)}_fp", 0) + fp
        out[f"g{int(g)}_fn"] = out.get(f"g{int(g)}_fn", 0) + fn
        out[f"g{int(g)}_n"] = out.get(f"g{int(g)}_n", 0) + n
    return out

def _rate(tp, fp, fn, tn, which: str) -> float:
    # TPR, FPR, PPV etc.
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    if which == "TPR":
        denom = tp + fn
        return tp/denom if denom else 0.0
    if which == "FPR":
        denom = fp + tn
        return fp/denom if denom else 0.0
    if which == "PR":
        denom = tp + fp
        return tp/denom if denom else 0.0
    raise ValueError(which)

def fairness_report(agg_counts: Dict[str, int]) -> Dict[str, Any]:
    # derive EO gap (TPR), FPR gap, Statistical parity (positive rate) across groups found in agg_counts
    # detect groups
    groups = sorted({int(k.split("_")[0][1:]) for k in agg_counts.keys() if k.startswith("g")})
    stats = {}
    tprs, fprs, prs, pos_rates = [], [], [], []
    for g in groups:
        tp, fp = agg_counts.get(f"g{g}_tp",0), agg_counts.get(f"g{g}_fp",0)
        fn, tn = agg_counts.get(f"g{g}_fn",0), agg_counts.get(f"g{g}_tn",0)
        n = agg_counts.get(f"g{g}_n",0)
        tpr = _rate(tp, fp, fn, tn, "TPR")
        fpr = _rate(tp, fp, fn, tn, "FPR")
        pr = _rate(tp, fp, fn, tn, "PR")
        pos = (tp + fp)/n if n else 0.0
        tprs.append(tpr); fprs.append(fpr); prs.append(pr); pos_rates.append(pos)
        stats[f"g{g}_TPR"]=tpr; stats[f"g{g}_FPR"]=fpr; stats[f"g{g}_PR"]=pr; stats[f"g{g}_PosRate"]=pos
    def gap(v): return max(v) - min(v) if v else 0.0
    stats["EO_gap"] = gap(tprs)
    stats["FPR_gap"] = gap(fprs)
    stats["SP_gap"] = gap(pos_rates)
    stats["max_group_gap"] = max(stats["EO_gap"], stats["FPR_gap"], stats["SP_gap"])
    return stats

def group_gap_penalty(logits: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Differentiable penalty: squared difference of per-group CE losses.
    """
    ce = F.cross_entropy
    groups = torch.unique(a)
    losses = []
    for g in groups:
        m = a == g
        if m.sum() == 0: continue
        losses.append(ce(logits[m], y[m]))
    if len(losses) < 2:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    L = torch.stack(losses)
    diffs = (L.unsqueeze(0) - L.unsqueeze(1))**2
    return diffs.mean()
