# faircare/core/client.py
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Back-compat shim: Algorithms like fedavg.py import ClientConfig from here.
# We re-introduce a simple dataclass so legacy imports work without touching algos.
# -----------------------------------------------------------------------------
@dataclass
class ClientConfig:
    batch_size: int = 64
    local_epochs: int = 1
    lr: float = 1e-3
    mu: float = 0.0             # for FedProx if needed
    lambdaG: float = 1.0        # group-fairness loss weight (if used)
    lambdaA: float = 0.5        # adversary loss weight (if used)
    use_pcgrad: bool = True     # conflict-aware training
    device: str = "cpu"

def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)

# -----------------------------------------------------------------------------
# Minimal PCGrad between two scalar losses (main and fairness/adversary)
# -----------------------------------------------------------------------------
def _pcgrad_two_losses(model: nn.Module, loss_main: torch.Tensor, loss_fair: torch.Tensor):
    """
    Projects fairness gradient to avoid destructive interference with main gradient.
    Reference: Yu et al., NeurIPS 2020 (PCGrad).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # grad of main
    torch.autograd.backward(loss_main, retain_graph=True)
    g_main = [ (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in params ]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # grad of fair
    torch.autograd.backward(loss_fair)
    g_fair = [ (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in params ]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # project if conflicting
    dot = sum((gm * gf).sum() for gm, gf in zip(g_main, g_fair))
    if dot < 0:
        norm2 = sum((gm ** 2).sum() for gm in g_main) + 1e-12
        g_fair = [ gf - (dot / norm2) * gm for gf, gm in zip(g_fair, g_main) ]

    # apply combined gradient
    for p, gm, gf in zip(params, g_main, g_fair):
        p.grad = gm + gf

# -----------------------------------------------------------------------------
# Local training loop with adversarial debiasing head and optional PCGrad
# -----------------------------------------------------------------------------
def local_train(model: nn.Module,
                adv_head: Optional[nn.Module],
                optimizer: torch.optim.Optimizer,
                X: np.ndarray, y: np.ndarray, s: np.ndarray,
                batch_size: int = 64, local_epochs: int = 1,
                lambdaG: float = 1.0, lambdaA: float = 0.5,
                use_pcgrad: bool = True,
                device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train a client model locally.

    Returns:
      report: {
        "loss": float,            # running loss proxy for weighting
        "gap": float,             # simple local DP-gap proxy (for weighting)
        "num_samples": int,
        "control_variate": None   # reserved for future SCAFFOLD-lite
      }
      deltas: { "delta": List[np.ndarray] }  # model parameter delta
    """
    model.to(device).train()
    if adv_head is not None:
        adv_head.to(device).train()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    s_t = torch.tensor(s, dtype=torch.long, device=device)

    ds = TensorDataset(X_t, y_t, s_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    init_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    running_loss = 0.0
    n_steps = 0

    for _ in range(local_epochs):
        for xb, yb, sb in dl:
            n_steps += 1
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            # main loss
            loss_main = ce(logits, yb)

            # fairness adversary on logits (predict sensitive attribute)
            if adv_head is not None and lambdaA > 0:
                adv_logits = adv_head(logits.detach())
                loss_adv = bce(adv_logits.squeeze(-1), sb.float())
                loss_fair = lambdaA * loss_adv
            else:
                loss_fair = torch.tensor(0.0, device=device)

            if use_pcgrad and loss_fair.requires_grad and loss_fair.item() > 0:
                _pcgrad_two_losses(model, loss_main, loss_fair)
                optimizer.step()
                total_loss = loss_main.item() + loss_fair.item()
            else:
                total = loss_main + loss_fair
                total.backward()
                optimizer.step()
                total_loss = float(total.detach().item())

            running_loss += total_loss

    # simple local DP-gap proxy for server weighting
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = (logits.softmax(1)[:, 1] > 0.5).long()
        pos1 = (preds[s_t == 1] == 1).float().mean().item() if (s_t == 1).sum() > 0 else 0.0
        pos0 = (preds[s_t == 0] == 1).float().mean().item() if (s_t == 0).sum() > 0 else 0.0
        dp_gap = abs(pos1 - pos0)

    final_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    delta = [f - i for f, i in zip(final_params, init_params)]

    report = {
        "loss": running_loss / max(1, n_steps),
        "gap": dp_gap,
        "num_samples": int(X_t.size(0)),
        "control_variate": None,
    }
    return report, {"delta": delta}

__all__ = ["ClientConfig", "make_optimizer", "local_train"]
