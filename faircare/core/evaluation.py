from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
from torch import nn
from .utils import Batch
from ..fairness.metrics import group_confusion_counts, fairness_report

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, sens_present: bool) -> Dict[str, Any]:
    model.eval()
    n, correct = 0, 0
    agg = {}
    for (x, y, a) in loader:
        batch = Batch(torch.as_tensor(x, dtype=torch.float32),
                      torch.as_tensor(y, dtype=torch.long),
                      torch.as_tensor(a, dtype=torch.long) if sens_present else None)
        batch = Batch(batch.x.to(device), batch.y.to(device), None if batch.a is None else batch.a.to(device))
        logits = model(batch.x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == batch.y).sum().item()
        n += batch.y.numel()
        if sens_present:
            # accumulate confusion counts by group
            counts = group_confusion_counts(pred.cpu(), batch.y.cpu(), batch.a.cpu())
            for k, v in counts.items():
                agg[k] = agg.get(k, 0) + v
    acc = correct / max(n, 1)
    report = fairness_report(agg) if sens_present else {}
    report["accuracy"] = acc
    return report
