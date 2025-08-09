# faircare/fairness/__init__.py
"""
Public API for fairness helpers.

We re-export the underscored implementations from metrics.py so that
older imports like `from faircare.fairness import dp_gap` continue to work.
"""

from .metrics import (
    _dp_gap as dp_gap,
    _eo_gap as eo_gap,
    _fpr_gap as fpr_gap,
    expected_calibration_error as calibration_ece,
    _confusion_by_group as confusion_by_group,
    compute_metrics,
    threshold_sweep,
)

__all__ = [
    "dp_gap",
    "eo_gap",
    "fpr_gap",
    "calibration_ece",
    "confusion_by_group",
    "compute_metrics",
    "threshold_sweep",
]
