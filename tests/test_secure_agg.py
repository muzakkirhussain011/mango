import torch
from faircare.core.secure_agg import SecureAggregator

def test_secure_agg_mean():
    agg = SecureAggregator()
    a = [torch.ones(5), 2*torch.ones(5), 3*torch.ones(5)]
    out = agg.aggregate(a)
    assert torch.allclose(out, torch.tensor([2.0]*5))
