# faircare/core/server.py
import copy, numpy as np, torch

class Server:
    def __init__(self, global_model, beta=0.9, device="cpu"):
        self.global_model = global_model
        self.beta = beta
        self.device = device
        self.momentum = {k: torch.zeros_like(v) for k,v in self.global_model.state_dict().items()}

    def broadcast(self, clients):
        st = self.global_model.state_dict()
        for c in clients:
            c.set_state(st)

    def _apply_delta(self, delta):
        new_state = copy.deepcopy(self.global_model.state_dict())
        for k in new_state:
            new_state[k] = new_state[k] + delta[k]
        self.global_model.load_state_dict(new_state)

    def aggregate(self, client_payloads, weights, use_momentum=True):
        base = self.global_model.state_dict()
        delta = {k: torch.zeros_like(v) for k,v in base.items()}
        for w, pl in zip(weights, client_payloads):
            sd = pl["state_dict"]
            for k in delta:
                delta[k] += w * (sd[k] - base[k])
        if use_momentum and self.beta>0:
            for k in delta:
                self.momentum[k] = self.beta * self.momentum[k] + (1-self.beta) * delta[k]
            self._apply_delta(self.momentum)
        else:
            self._apply_delta(delta)
