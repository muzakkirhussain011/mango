# faircare/core/client.py
from typing import Any, Dict
import torch
import torch.nn as nn
from .utils import to_device
from ..fairness.metrics import compute_group_counts_from_batch

class Client:
    def __init__(self, model_ctor, adversary_ctor=None, device="cpu", sensitive_key="a"):
        self.model = model_ctor().to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.sensitive_key = sensitive_key
        self.adversary = adversary_ctor().to(device) if adversary_ctor is not None else None

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    def local_train(self, loader, epochs: int, lr: float, weight_decay: float,
                    fairness_weights: Dict[str, float], sens_present: bool) -> Dict[str, Any]:
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        adv_opt = torch.optim.Adam(self.adversary.parameters(), lr=lr) if self.adversary is not None else None

        lambda_eo = fairness_weights.get("lambda_eo", 0.0)
        lambda_sp = fairness_weights.get("lambda_sp", 0.0)

        total = 0
        fairness_counts = {}
        for _ in range(epochs):
            for batch in loader:
                x, y, a = to_device(batch["x"], self.device), to_device(batch["y"], self.device), \
                          to_device(batch.get(self.sensitive_key, None), self.device)

                logits = self.model(x).squeeze(-1)
                base_loss_vec = self.criterion(logits, y.float())
                # inverse-frequency reweighting by group to raise minority gradient
                if sens_present and a is not None:
                    # simple 2-group case (0/1). Extendable.
                    mask0 = (a == 0)
                    mask1 = (a == 1)
                    n0 = mask0.sum().clamp_min(1)
                    n1 = mask1.sum().clamp_min(1)
                    w0 = (n0 + n1).float() / (2.0 * n0.float())
                    w1 = (n0 + n1).float() / (2.0 * n1.float())
                    weights = torch.where(mask0, w0, w1)
                else:
                    weights = torch.ones_like(base_loss_vec)

                base_loss = (weights * base_loss_vec).mean()

                # optional adversary to remove sensitive leakage (if present)
                adv_loss = 0.0
                if self.adversary is not None and sens_present and a is not None:
                    with torch.no_grad():
                        h = torch.sigmoid(logits).detach()
                    p_a = self.adversary(h.unsqueeze(-1))
                    adv_crit = nn.BCEWithLogitsLoss()
                    adv_loss = adv_crit(p_a.squeeze(-1), a.float())

                # fairness surrogates: encourage equal TPR (EO) and parity (SP) by penalizing
                # group loss differences (simple, stable proxy)
                if sens_present and a is not None:
                    loss0 = base_loss_vec[a == 0].mean() if (a == 0).any() else 0.0
                    loss1 = base_loss_vec[a == 1].mean() if (a == 1).any() else 0.0
                    eo_proxy = (loss0 - loss1).abs()
                    sp_proxy = (torch.sigmoid(logits)[a == 0].mean() - torch.sigmoid(logits)[a == 1].mean()).abs()
                else:
                    eo_proxy = torch.tensor(0.0, device=self.device)
                    sp_proxy = torch.tensor(0.0, device=self.device)

                total_loss = base_loss + lambda_eo * eo_proxy + lambda_sp * sp_proxy
                if self.adversary is not None:
                    total_loss = total_loss - 0.1 * adv_loss  # confuse adversary

                opt.zero_grad()
                if self.adversary is not None:
                    adv_opt.zero_grad()
                total_loss.backward()
                opt.step()
                if self.adversary is not None:
                    # adversary maximizes ability to predict a from h
                    with torch.no_grad():
                        h = torch.sigmoid(self.model(x).squeeze(-1)).detach()
                    p_a = self.adversary(h.unsqueeze(-1))
                    adv_crit = nn.BCEWithLogitsLoss()
                    adv_loss = adv_crit(p_a.squeeze(-1), a.float())
                    adv_loss.backward()
                    adv_opt.step()

                # bookkeeping
                total += x.size(0)
                # streaming fairness counts for post-processing
                batch_counts = compute_group_counts_from_batch(logits.detach(), y, a)
                # merge
                for k, v in batch_counts.items():
                    fairness_counts[k] = fairness_counts.get(k, 0) + v

        return {
            "state_dict": self.model.state_dict(),
            "num_samples": total,
            "fairness_counts": fairness_counts
        }
