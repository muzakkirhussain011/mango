# faircare/core/evaluation.py
from typing import Dict, Any
import torch
from .utils import to_device
from ..fairness.summarize import summarize_by_group
from ..fairness.postprocess import equalized_odds_thresholds_from_counts, apply_group_thresholds

@torch.no_grad()
def evaluate_global(model, loader, device="cpu", sens_present=True, apply_postprocess=True) -> Dict[str, float]:
    model.eval()
    ys, ps, asens = [], [], []
    for batch in loader:
        x, y = to_device(batch["x"], device), to_device(batch["y"], device)
        logits = model(x).squeeze(-1)
        p = torch.sigmoid(logits)
        ys.append(y.cpu())
        ps.append(p.cpu())
        if sens_present and "a" in batch:
            asens.append(batch["a"].cpu())
    y = torch.cat(ys)
    p = torch.cat(ps)
    a = torch.cat(asens) if sens_present and len(asens) > 0 else None

    metrics = summarize_by_group(p, y, a)

    if apply_postprocess and a is not None:
        # derive thresholds from counts (we reuse summarize which bins internally)
        th = equalized_odds_thresholds_from_counts(p, y, a)
        yhat_pp = apply_group_thresholds(p, a, th)
        metrics_pp = summarize_by_group(yhat_pp.float(), y, a, already_binary=True)
        # prefer EO-improved metrics if no utility drop > 0.5%
        if (metrics_pp["EO_gap"] + 1e-6) < metrics["EO_gap"] and \
           (metrics_pp["accuracy"] + 1e-6) >= metrics["accuracy"] - 0.005:
            metrics = metrics_pp
            metrics["postprocess"] = 1.0
        else:
            metrics["postprocess"] = 0.0
    else:
        metrics["postprocess"] = 0.0

    return metrics

