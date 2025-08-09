# faircare/data/synth_health.py
import numpy as np

def make_synth(n=4000, d=20, sens_ratio=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n,d)).astype("float32")
    s = rng.binomial(1, sens_ratio, size=n).astype(int)
    w = rng.normal(size=(d,))
    logits = X @ w + 0.8*(s==1) - 0.2*(s==0)
    y = (logits + rng.normal(scale=0.5, size=n) > 0.0).astype(int)
    return X, y, s, [f"x{i}" for i in range(d)]
