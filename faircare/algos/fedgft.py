# faircare/algos/fedgft.py
from dataclasses import dataclass
from ..core.client import ClientConfig
from .aggregator import weights_fedavg

@dataclass
class FedGFT:
    local_epochs: int
    batch_size: int
    lr: float
    lambdaG: float = 2.0
    def client_cfg(self):
        return ClientConfig(self.local_epochs, self.batch_size, self.lr, self.lambdaG, 0.0, 0.0, use_adversary=True)
    def compute_weights(self, payloads):
        return weights_fedavg(payloads)
    def use_momentum(self): return False
