# faircare/algos/qffl.py
from dataclasses import dataclass
from ..core.client import ClientConfig
from .aggregator import weights_qffl

@dataclass
class QFFL:
    local_epochs: int
    batch_size: int
    lr: float
    q: float
    def client_cfg(self):
        return ClientConfig(self.local_epochs, self.batch_size, self.lr, 0.0, 0.5, self.q, use_adversary=False)
    def compute_weights(self, payloads):
        return weights_qffl(payloads, self.q)
    def use_momentum(self): return False
