"""Fairness metrics and utilities."""

from faircare.fairness.metrics import (
    group_confusion_counts,
    fairness_report,
    compute_group_metrics
)

__all__ = [
    "group_confusion_counts",
    "fairness_report",
    "compute_group_metrics"
]
