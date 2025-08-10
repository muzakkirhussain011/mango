from __future__ import annotations
import numpy as np

def make_synth_health(n=10000, p=20, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(int)
    w = rng.normal(size=(p, ))
    logits = X @ w + (a * 0.8)  # induced bias
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > 0.5).astype(int)
    return X, y, a
