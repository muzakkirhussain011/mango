# faircare/fairness/__init__.py
from .metrics import dp_gap, eo_gap, fpr_gap, calibration_ece, confusion_by_group
from .global_stats import aggregate_group_stats
