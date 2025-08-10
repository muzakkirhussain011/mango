from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ..models.classifier import MLPClassifier
from ..models.adversary import AdversaryLoss
from ..fairness.metrics import group_gap_penalty
from .utils import optimizer_for, Batch, to_device

class Client:
    def __init__(
        self,
        cid: int,
        model: nn.Module,
        device: torch.device,
        lambda_g: float,
        lambda_c: float,
        fairness_mode: str = "penalty",
        qffl_q: float = 0.0,
        fedprox_mu: float = 0.0,
        reference_model: Optional[nn.Module] = None,
    ):
        self.cid = cid
        self.model = model
        self.device = device
        self.lambda_g = lambda_g
        self.lambda_c = lambda_c
        self.fairness_mode = fairness_mode
        self.qffl_q = qffl_q
        self.fedprox_mu = fedprox_mu
        self.reference_model = reference_model
        self.adv = AdversaryLoss() if fairness_mode == "adversarial" else None

    def local_train(
        self,
        data_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        sensitive_present: bool,
        client_weight_factor: float = 1.0,
    ) -> Dict[str, Any]:
        self.model.train()
        opt = optimizer_for(self.model, lr, weight_decay)
        ref_params = {k: v.detach().clone() for k, v in self.reference_model.state_dict().items()} if self.reference_model else None
        ce = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for (x, y, a) in data_loader:
                batch = Batch(torch.as_tensor(x, dtype=torch.float32),
                              torch.as_tensor(y, dtype=torch.long),
                              torch.as_tensor(a, dtype=torch.long) if sensitive_present else None)
                batch = to_device(batch, self.device)
                logits = self.model(batch.x)
                base_loss = ce(logits, batch.y)

                # q-FFL weighting (loss^(q) scaling)
                if self.qffl_q > 0:
                    base_loss = base_loss * (base_loss.detach() + 1e-12) ** self.qffl_q

                loss = base_loss

                # FedProx proximal term
                if self.fedprox_mu > 0 and self.reference_model is not None:
                    prox = 0.0
                    for (n, p) in self.model.named_parameters():
                        prox += torch.norm(p - ref_params[n]) ** 2
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                # Group fairness regularizer
                if self.lambda_g > 0 and sensitive_present:
                    if self.fairness_mode == "penalty":
                        # differentiable penalty on per-group loss disparity
                        gpen = group_gap_penalty(logits, batch.y, batch.a)
                        loss = loss + self.lambda_g * gpen
                    else:
                        # adversarial debiasing via gradient reversal
                        loss = loss + self.lambda_g * self.adv(logits, batch.a)

                # Client "importance" proxy (align with server fairness signal)
                if self.lambda_c > 0:
                    loss = loss * (1.0 + self.lambda_c * (client_weight_factor - 1.0))

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Collect local stats for fairness aggregation
        with torch.no_grad():
            deltas = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        return {"cid": self.cid, "weights": deltas}
