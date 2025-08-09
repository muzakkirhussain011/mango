# faircare/algos/faircare_fl.py
from dataclasses import dataclass
from ..core.client import ClientConfig
from .aggregator import weights_faircare, normalize_weights, weights_afl

@dataclass
class FairCareFL:
    local_epochs: int
    batch_size: int
    lr: float
    lambdaG: float
    lambdaC: float
    lambdaA: float
    q: float
    beta: float
    use_adv: bool = True

    def client_cfg(self):
        return ClientConfig(self.local_epochs, self.batch_size, self.lr, self.lambdaG, self.lambdaC, self.q, use_adversary=self.use_adv)

    def compute_weights(self, payloads):
        w_base = weights_faircare(payloads, self.q)
        if self.lambdaA > 0:
            w_afl = weights_afl(payloads, boost=1.0 + 4.0*self.lambdaA)
            w = [0.7*wb + 0.3*wa for wb,wa in zip(w_base, w_afl)]
            return normalize_weights(w)
        return w_base

    def use_momentum(self): return True if self.beta>0 else False
