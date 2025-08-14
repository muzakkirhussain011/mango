"""Test fairness metrics."""

import pytest
import torch
import numpy as np
from faircare.fairness.metrics import group_confusion_counts, fairness_report


class TestMetrics:
    
    def test_group_confusion_counts(self):
        """Test confusion matrix computation."""
        # Create test data
        y_pred = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0])
        y_true = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0])
        a = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Compute counts
        counts = group_confusion_counts(y_true, y_pred, a)  # Note: y_true, y_pred order
        
        # IMPORTANT: The implementation maps sensitive=1 to group_0 and sensitive=0 to group_1
        # So group_0 corresponds to a=1 (indices 4-7) and group_1 to a=0 (indices 0-3)
        
        # Check group_0 (a=1, indices 4-7)
        # Index 4: y_pred=0, y_true=0 → TN
        # Index 5: y_pred=1, y_true=1 → TP
        # Index 6: y_pred=0, y_true=1 → FN
        # Index 7: y_pred=0, y_true=0 → TN
        assert counts["group_0"]["TP"] == 1
        assert counts["group_0"]["TN"] == 2
        assert counts["group_0"]["FP"] == 0
        assert counts["group_0"]["FN"] == 1
        
        # Check group_1 (a=0, indices 0-3)
        # Index 0: y_pred=1, y_true=1 → TP
        # Index 1: y_pred=0, y_true=0 → TN
        # Index 2: y_pred=1, y_true=0 → FP
        # Index 3: y_pred=1, y_true=1 → TP
        assert counts["group_1"]["TP"] == 2
        assert counts["group_1"]["TN"] == 1
        assert counts["group_1"]["FP"] == 1
        assert counts["group_1"]["FN"] == 0
    
    def test_fairness_report(self):
        """Test fairness report generation."""
        # Create test data with known properties
        y_pred = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
        y_true = torch.tensor([1, 0, 0, 1, 1, 0, 0, 1])
        a = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Generate report
        report = fairness_report(y_pred, y_true, a)
        
        # Check basic metrics
        assert "accuracy" in report
        assert "macro_F1" in report
        
        # Check group metrics
        assert "g0_TPR" in report
        assert "g1_TPR" in report
        
        # Check fairness gaps
        assert "EO_gap" in report
        assert "FPR_gap" in report
        assert "SP_gap" in report
        assert "max_group_gap" in report
        
        # EO gap should be |TPR_0 - TPR_1|
        eo_gap = abs(report["g0_TPR"] - report["g1_TPR"])
        assert abs(report["EO_gap"] - eo_gap) < 1e-6
    
    def test_fairness_report_no_sensitive(self):
        """Test fairness report without sensitive attribute."""
        y_pred = torch.tensor([1, 0, 1, 0])
        y_true = torch.tensor([1, 0, 0, 0])
        
        report = fairness_report(y_pred, y_true, None)
        
        # Should have accuracy but gaps should be 0
        assert report["accuracy"] == 0.75
        assert report["EO_gap"] == 0.0
        assert report["FPR_gap"] == 0.0
        assert report["SP_gap"] == 0.0
