# faircare/fairness/summarize.py
import numpy as np
from .metrics import dp_gap, eo_gap, fpr_gap, calibration_ece, confusion_by_group

def summarize(y_true, y_prob, y_pred, sensitive):
    return {
        "dp_gap": dp_gap(y_true, y_pred, sensitive),
        "eo_gap": eo_gap(y_true, y_pred, sensitive),
        "fpr_gap": fpr_gap(y_true, y_pred, sensitive),
        "ece": calibration_ece(y_true, y_prob),
        "conf_by_group": confusion_by_group(y_true, y_pred, sensitive),
    }
