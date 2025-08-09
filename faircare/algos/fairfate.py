# faircare/algos/fairfate.py
from dataclasses import dataclass
from ..core.client import ClientConfig
from .aggregator import weights_fairfed

@dataclass
class FairFATE:
    local_epochs: int
    batch_size: int
    lr: float
    def client_cfg(self):
        return ClientConfig(self.local_epochs, self.batch_size, self.lr, 1.0, 0.0, 0.0, use_adversary=True)
    def compute_weights(self, payloads):
        return weights_fairfed(payloads)
    def use_momentum(self): return True
