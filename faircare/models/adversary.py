# faircare/models/adversary.py
import torch, torch.nn as nn

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class LogitAdversary(nn.Module):
    """Adversary on logits to predict sensitive attribute; used with GradReverse."""
    def __init__(self, in_dim: int = 2, hidden: int = 16, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, logits, lambd: float = 1.0):
        x = GradReverse.apply(logits, lambd)
        return self.net(x)
