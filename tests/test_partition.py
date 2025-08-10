from faircare.data.partition import dirichlet_partition
import numpy as np

def test_dirichlet_partition_covers_all():
    y = np.array([0,1,0,1,0,1,0,1,1,0])
    parts = dirichlet_partition(y, n_clients=3, alpha=0.5)
    covered = sorted(np.concatenate(parts).tolist())
    assert covered == list(range(len(y)))
