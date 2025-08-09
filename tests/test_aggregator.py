# tests/test_aggregator.py
import numpy as np
from faircare.algos.aggregator import normalize_weights, weights_qffl

def test_normalize():
    w = normalize_weights([0.5, 0.5])
    assert abs(sum(w) - 1.0) < 1e-6

def test_qffl_monotonicity():
    payloads = [{"val_loss": 0.2}, {"val_loss": 1.0}]
    w_lowq = weights_qffl(payloads, q=0.1)
    w_highq = weights_qffl(payloads, q=1.0)
    assert w_highq[1] >= w_lowq[1]
