# faircare/core/client.py
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _pcgrad_two_losses(model: nn.Module, loss_main: torch.Tensor, loss_fair: torch.Tensor):
    """
    Minimal PCGrad for two objectives.
    Projects fairness gradient to avoid conflict with main gradient (Yu et al., NeurIPS'20).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    g_list = []

    # grad of main
    torch.autograd.backward(loss_main, retain_graph=True)
    g_main = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # grad of fair
    torch.autograd.backward(loss_fair)
    g_fair = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # project fairness grad if conflicting
    dot = sum((gm * gf).sum() for gm, gf in zip(g_main, g_fair))
    if dot < 0:
        # project gf onto plane orthogonal to gm
        norm2 = sum((gm ** 2).sum() for gm in g_main) + 1e-12
        g_fair = [gf - (dot / norm2) * gm for gf, gm in zip(g_fair, g_main)]

    # apply combined gradient (sum)
    for p, gm, gf in zip(params, g_main, g_fair):
        p.grad = gm + gf

def local_train(model: nn.Module,
                adv_head: nn.Module,
                optimizer: torch.optim.Optimizer,
                X: np.ndarray, y: np.ndarray, s: np.ndarray,
                batch_size: int = 64, local_epochs: int = 1,
                lambdaG: float = 1.0, lambdaA: float = 0.5,
                use_pcgrad: bool = True,
                device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    One client's local training step with adversarial debiasing (GRL head) and optional PCGrad.
    Returns:
      report: {"loss": float, "gap": float, "num_samples": int, "control_variate": Optional[List[np.ndarray]]}
      deltas: { "delta": List[np.ndarray] }
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
    n = 0
    running_loss, running_gap = 0.0, 0.0

    for _ in range(local_epochs):
        for xb, yb, sb in dl:
            n += xb.size(0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            # main task loss
            loss_main = ce(logits, yb)

            # adversarial fairness loss on logits (predict sensitive attr)
            if adv_head is not None and lambdaA > 0:
                with torch.enable_grad():
                    adv_logits = adv_head(logits.detach())  # detach features; GRL is simulated by sign flip
                    loss_adv = bce(adv_logits.squeeze(-1), sb.float())
                loss_fair = lambdaA * loss_adv
            else:
                loss_fair = torch.tensor(0.0, device=device)

            if use_pcgrad and loss_fair.requires_grad and loss_fair.item() > 0:
                _pcgrad_two_losses(model, loss_main, loss_fair)
                optimizer.step()
                total_loss = loss_main.item() + loss_fair.item()
            else:
                total_loss = loss_main + loss_fair
                total_loss.backward()
                optimizer.step()
                total_loss = float(total_loss.detach().item())

            running_loss += total_loss

    # simple local DP gap on final predictions (rough proxy for weighting)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = (logits.softmax(1)[:, 1] > 0.5).long()
        pos_a = (preds[s_t == 1] == 1).float().mean().item() if (s_t == 1).sum() > 0 else 0.0
        pos_b = (preds[s_t == 0] == 1).float().mean().item() if (s_t == 0).sum() > 0 else 0.0
        dp_gap = abs(pos_a - pos_b)

    final_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    delta = [f - i for f, i in zip(final_params, init_params)]

    report = {
        "loss": running_loss / max(1, len(dl) * local_epochs),
        "gap": dp_gap,
        "num_samples": int(len(X_t)),
        "control_variate": None  # reserved for SCAFFOLD-lite
    }
    return report, {"delta": delta}
