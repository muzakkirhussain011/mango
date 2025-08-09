# tests/test_theory_checks.py
import numpy as np
from faircare.algos.aggregator import weights_faircare

def test_faircare_weighting_prefers_fair_clients():
    payloads = [
        {"val_loss": 0.5, "summary":{"dp_gap":0.2,"eo_gap":0.2}},
        {"val_loss": 0.5, "summary":{"dp_gap":0.0,"eo_gap":0.0}}
    ]
    w = weights_faircare(payloads, q=0.5)
    assert w[1] > w[0]
