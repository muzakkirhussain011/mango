# faircare/algos/aggregator.py
import numpy as np

def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    return w.tolist()

def weights_fedavg(payloads):
    n = len(payloads)
    return normalize_weights([1.0/n]*n)

def weights_qffl(payloads, q: float):
    losses = [max(1e-6, pl["val_loss"]) for pl in payloads]
    raw = [l**q for l in losses]
    return normalize_weights(raw)

def weights_fairfed(payloads):
    gaps = [pl["summary"]["dp_gap"] + pl["summary"]["eo_gap"] for pl in payloads]
    raw = [1.0/max(1e-6, g) for g in gaps]
    return normalize_weights(raw)

def weights_afl(payloads, boost=3.0):
    losses = [pl["val_loss"] for pl in payloads]
    worst = int(np.argmax(losses))
    w = np.ones(len(payloads))
    w[worst] = w[worst]*boost
    return normalize_weights(w)

def weights_faircare(payloads, q: float):
    losses = [max(1e-6, pl["val_loss"]) for pl in payloads]
    gaps = [pl["summary"]["dp_gap"] + pl["summary"]["eo_gap"] for pl in payloads]
    raw = [(l**q)/(g+1e-6) for l,g in zip(losses, gaps)]
    return normalize_weights(raw)
