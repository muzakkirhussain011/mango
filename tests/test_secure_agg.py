# tests/test_secure_agg.py
import numpy as np
from faircare.core.secure_agg import mask_vector, unmask_sum

def test_mask_unmask_sum():
    rng = np.random.default_rng(0)
    vs = []
    masks = []
    raw = []
    for _ in range(5):
        v = rng.integers(0, 10, size=8, dtype=np.int64)
        mv, m = mask_vector(v, rng)
        vs.append(mv); masks.append(m); raw.append(v)
    rec = unmask_sum(np.stack(vs), np.stack(masks))
    assert np.all(rec == np.sum(np.stack(raw), axis=0))
