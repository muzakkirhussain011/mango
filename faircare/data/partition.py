from __future__ import annotations
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from .adult import load_adult
from .heart import load_heart
from .mimic_eicu import load_mimic_eicu
from .synth_health import make_synth_health

def _load(dataset: str, cache_dir: str, sensitive: str):
    if dataset == "adult":
        return load_adult(cache_dir, sensitive)
    if dataset == "heart":
        return load_heart(cache_dir, sensitive)
    if dataset == "mimic_eicu":
        return load_mimic_eicu(cache_dir, sensitive)
    if dataset == "synth_health":
        return make_synth_health()
    raise ValueError(f"Unknown dataset {dataset}")

def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    """
    Create indices per client using Dirichlet prior on class proportions.
    """
    n = len(y); n_classes = len(np.unique(y))
    idx_by_class = [np.where(y == c)[0] for c in range(n_classes)]
    props = np.random.dirichlet(alpha=np.ones(n_clients)*alpha, size=n_classes)
    client_ids = [[] for _ in range(n_clients)]
    for c, idxs in enumerate(idx_by_class):
        np.random.shuffle(idxs)
        splits = (props[c] / props[c].sum() * len(idxs)).astype(int)
        # ensure sum equals len(idxs)
        diff = len(idxs) - splits.sum()
        splits[0] += diff
        start = 0
        for i in range(n_clients):
            take = splits[i]
            client_ids[i].extend(idxs[start:start+take])
            start += take
    return [np.array(sorted(cid)) for cid in client_ids]

def make_federated_splits(dataset: str, n_clients: int, alpha: float, sensitive: str, cache_dir: str):
    X, y, a = _load(dataset, cache_dir, sensitive)
    X_tr, X_va, y_tr, y_va, a_tr, a_va = train_test_split(X, y, a, test_size=0.2, stratify=y, random_state=42)
    parts = dirichlet_partition(y_tr, n_clients, alpha)
    train_parts = []
    for idx in parts:
        train_parts.append((X_tr[idx], y_tr[idx], a_tr[idx]))
    input_dim = X.shape[1]
    n_classes = len(np.unique(y))
    sens_present = a is not None
    return train_parts, (X_va, y_va, a_va), input_dim, n_classes, sens_present
