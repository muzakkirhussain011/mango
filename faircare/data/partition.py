# faircare/data/partition.py
import numpy as np
from typing import List, Union, Sequence

def dirichlet_partition(
    data_or_n: Union[int, Sequence],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[np.ndarray]:
    """Partition indices using a Dirichlet distribution.

    Parameters
    ----------
    data_or_n:
        Either the dataset/labels to be partitioned or the number of samples.
    num_clients:
        Number of client partitions to produce.
    alpha:
        Dirichlet concentration parameter controlling non-IID severity.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List[np.ndarray]
        A list of index arrays, one per client.
    """

    rng = np.random.default_rng(seed)

    if isinstance(data_or_n, (int, np.integer)):
        n = int(data_or_n)
    else:  # assume sequence/array-like
        n = len(data_or_n)

    idx = np.arange(n)
    rng.shuffle(idx)
    props = rng.dirichlet(alpha=[alpha] * num_clients, size=n)
    buckets = [[] for _ in range(num_clients)]
    for i, p in enumerate(props):
        c = rng.choice(num_clients, p=p)
        buckets[c].append(idx[i])
    return [np.array(b, dtype=int) for b in buckets]
