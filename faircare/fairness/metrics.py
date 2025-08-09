# faircare/fairness/metrics.py
import numpy as np

def _by_group(y_true, y_pred, y_prob, sensitive):
    groups = np.unique(sensitive)
    stats = {}
    for g in groups:
        mask = (sensitive == g)
        yt, yp = y_true[mask], y_pred[mask]
        pp = y_prob[mask]
        if len(yt)==0:
            continue
        tp = ((yp==1) & (yt==1)).sum()
        fp = ((yp==1) & (yt==0)).sum()
        fn = ((yp==0) & (yt==1)).sum()
        tn = ((yp==0) & (yt==0)).sum()
        sr = (yp==1).mean()
        tpr = tp / max(1, (tp+fn))
        fpr = fp / max(1, (fp+tn))
        stats[int(g)] = {"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"sr":float(sr),
                         "tpr":float(tpr),"fpr":float(fpr),"n":int(mask.sum())}
    return stats

def dp_gap(y_true, y_pred, sensitive):
    stats = _by_group(y_true, y_pred, y_pred, sensitive)
    srs = [v["sr"] for v in stats.values()]
    return float(max(srs)-min(srs)) if srs else 0.0

def eo_gap(y_true, y_pred, sensitive):
    stats = _by_group(y_true, y_pred, y_pred, sensitive)
    tprs = [v["tpr"] for v in stats.values()]
    return float(max(tprs)-min(tprs)) if tprs else 0.0

def fpr_gap(y_true, y_pred, sensitive):
    stats = _by_group(y_true, y_pred, y_pred, sensitive)
    fprs = [v["fpr"] for v in stats.values()]
    return float(max(fprs)-min(fprs)) if fprs else 0.0

def calibration_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0., 1., n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum()==0: continue
        acc = (y_true[m] == (y_prob[m] >= 0.5)).mean()
        conf = y_prob[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

def confusion_by_group(y_true, y_pred, sensitive):
    stats = _by_group(y_true, y_pred, y_pred, sensitive)
    return {g: {k:int(v[k]) if isinstance(v[k], (int, np.integer)) else float(v[k])
                for k in ("tp","fp","fn","tn","n")} for g,v in stats.items()}
