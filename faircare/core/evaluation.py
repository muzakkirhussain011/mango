# faircare/core/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def overall_metrics(y_true, y_prob, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "auroc": auc}
