# faircare/fairness/metrics.py
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

def _confusion_by_group(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray):
    out = {}
    for g in np.unique(s):
        m = s == g
        yt, yp = y_true[m], y_pred[m]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        out[int(g)] = (tp, fp, tn, fn)
    return out

def _dp_gap(y_pred: np.ndarray, s: np.ndarray):
    groups = np.unique(s)
    rates = []
    for g in groups:
        m = s == g
        rates.append((y_pred[m] == 1).mean() if m.any() else 0.0)
    return float(np.abs(rates[0] - rates[1])) if len(rates) == 2 else float(np.max(rates) - np.min(rates))

def _eo_gap(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray):
    groups = np.unique(s)
    tprs = []
    for g in groups:
        m = s == g
        yt, yp = y_true[m], y_pred[m]
        pos = (yt == 1)
        tpr = ( (yp[pos] == 1).mean() if pos.any() else 0.0 )
        tprs.append(tpr)
    return float(np.abs(tprs[0] - tprs[1])) if len(tprs) == 2 else float(np.max(tprs) - np.min(tprs))

def _fpr_gap(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray):
    groups = np.unique(s)
    fprs = []
    for g in groups:
        m = s == g
        yt, yp = y_true[m], y_pred[m]
        neg = (yt == 0)
        fpr = ( (yp[neg] == 1).mean() if neg.any() else 0.0 )
        fprs.append(fpr)
    return float(np.abs(fprs[0] - fprs[1])) if len(fprs) == 2 else float(np.max(fprs) - np.min(fprs))

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.any():
            conf = y_prob[m].mean()
            acc  = (y_true[m] == (y_prob[m] >= 0.5)).mean()
            ece += (m.mean()) * abs(acc - conf)
    return float(ece)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = (y_pred == y_true).mean()
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    dp = _dp_gap(y_pred, s)
    eo = _eo_gap(y_true, y_pred, s)
    fpr = _fpr_gap(y_true, y_pred, s)
    ece = expected_calibration_error(y_true, y_prob)
    return {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "dp_gap": float(dp),
        "eo_gap": float(eo),
        "fpr_gap": float(fpr),
        "ece": float(ece),
    }

def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, s: np.ndarray) -> Dict[str, Any]:
    """
    Sweep thresholds in [0.05, 0.95], pick:
      - best_acc: highest accuracy
      - best_eo: minimal EO gap
      - best_dp: minimal DP gap
    Returns dict of metrics per pick to visualize fairness/utility trade-offs.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    results = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, s, threshold=t)
        results.append((t, m))
    best_acc = max(results, key=lambda tm: tm[1]["accuracy"])
    best_eo = min(results, key=lambda tm: tm[1]["eo_gap"])
    best_dp = min(results, key=lambda tm: tm[1]["dp_gap"])
    return {
        "best_acc": {"threshold": float(best_acc[0]), **best_acc[1]},
        "best_eo":  {"threshold": float(best_eo[0]),  **best_eo[1]},
        "best_dp":  {"threshold": float(best_dp[0]),  **best_dp[1]},
    }
