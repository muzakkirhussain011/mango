# tests/test_metrics.py
import numpy as np
from faircare.fairness.metrics import dp_gap, eo_gap, fpr_gap, calibration_ece

def test_gaps_symmetric():
    y_true = np.array([0,0,1,1,0,1,0,1])
    y_pred = np.array([0,1,1,1,0,0,0,1])
    s = np.array([0,0,0,0,1,1,1,1])
    assert dp_gap(y_true, y_pred, s) >= 0.0
    assert eo_gap(y_true, y_pred, s) >= 0.0
    assert fpr_gap(y_true, y_pred, s) >= 0.0

def test_calibration_range():
    y_true = np.array([0,1,0,1,0,1,0,1])
    y_prob = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6])
    ece = calibration_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
