# faircare/data/partition.py
import numpy as np
from typing import List

def dirichlet_partition(n: int, num_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    props = rng.dirichlet(alpha=[alpha]*num_clients, size=n)
    buckets = [[] for _ in range(num_clients)]
    for i, p in enumerate(props):
        c = rng.choice(num_clients, p=p)
        buckets[c].append(idx[i])
    return [np.array(b, dtype=int) for b in buckets]
