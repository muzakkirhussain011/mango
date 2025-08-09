# tests/test_partition.py
import numpy as np
from faircare.data.partition import dirichlet_partition

def test_dirichlet_partition_covers_all():
    n = 100
    cl = 5
    parts = dirichlet_partition(n, cl, alpha=0.5, seed=123)
    total = sum(len(p) for p in parts)
    assert total == n
    assert len(parts) == cl
