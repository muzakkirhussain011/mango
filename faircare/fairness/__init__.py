# faircare/fairness/__init__.py
"""
Public API for fairness utilities.
Re-exports flexible fairness_report and group_confusion_counts for tests.
"""

from .metrics import group_confusion_counts, fairness_report

# Optional trackers if your code uses them; harmless if absent in tests
try:
    from .global_stats import FairnessTracker, RunningStats  # type: ignore
    __all__ = [
        "group_confusion_counts",
        "fairness_report",
        "FairnessTracker",
        "RunningStats",
    ]
except Exception:
    __all__ = [
        "group_confusion_counts",
        "fairness_report",
    ]
