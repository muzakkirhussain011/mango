import torch
from faircare.fairness.metrics import group_confusion_counts, fairness_report

def test_group_metrics():
    y = torch.tensor([0,0,1,1,0,1,1,0])
    p = torch.tensor([0,1,1,1,0,0,1,0])
    a = torch.tensor([0,0,0,1,1,1,1,1])
    counts = group_confusion_counts(p, y, a)
    rep = fairness_report(counts)
    assert "EO_gap" in rep and "SP_gap" in rep
