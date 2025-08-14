from typing import Dict, List, Optional, Tuple, Any, Union, Protocol

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
        counts = group_confusion_counts(y_pred, y_true, a)
        
        # Check group 0
        assert counts["group_0"]["TP"] == 2  # Both 1s correctly predicted
        assert counts["group_0"]["TN"] == 2  # Both 0s correctly predicted
        assert counts["group_0"]["FP"] == 1  # One 0 predicted as 1
        assert counts["group_0"]["FN"] == 0  # No 1s predicted as 0
        
        # Check group 1
        assert counts["group_1"]["TP"] == 1
        assert counts["group_1"]["TN"] == 2
        assert counts["group_1"]["FP"] == 0
        assert counts["group_1"]["FN"] == 1
    
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
