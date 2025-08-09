# faircare/core/client.py
from dataclasses import dataclass
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ..models.classifier import MLPClassifier
from ..models.adversary import LogitAdversary
from ..fairness.summarize import summarize

@dataclass
class ClientConfig:
    local_epochs: int
    batch_size: int
    lr: float
    lambdaG: float
    lambdaC: float
    q: float
    use_adversary: bool = True

class Client:
    def __init__(self, cid, X, y, s, X_val, y_val, s_val, input_dim, device="cpu"):
        self.cid = cid
        self.device = device
        self.dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(s, dtype=torch.long),
        )
        self.val = (torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                    torch.tensor(s_val, dtype=torch.long))
        self.model = MLPClassifier(input_dim).to(device)
        self.adv = LogitAdversary(in_dim=2).to(device)

    def set_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def train_local(self, cfg: ClientConfig):
        self.model.train()
        self.adv.train()
        loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)
        optM = optim.Adam(self.model.parameters(), lr=cfg.lr)
        optA = optim.Adam(self.adv.parameters(), lr=cfg.lr)
        cel = nn.CrossEntropyLoss()

        for _ in range(cfg.local_epochs):
            for xb, yb, sb in loader:
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)
                # adversary step
                if cfg.use_adversary and cfg.lambdaG>0:
                    with torch.no_grad():
                        logits = self.model(xb)
                    adv_logits = self.adv(logits.detach(), lambd=1.0)
                    loss_adv = cel(adv_logits, sb)
                    optA.zero_grad(); loss_adv.backward(); optA.step()

                # classifier step
                logits = self.model(xb)
                loss_task = cel(logits, yb)
                loss = loss_task
                if cfg.use_adversary and cfg.lambdaG>0:
                    adv_logits = self.adv(logits, lambd=1.0)
                    loss_fair = cel(adv_logits, sb)
                    loss = loss + (-cfg.lambdaG) * loss_fair
                optM.zero_grad(); loss.backward(); optM.step()

        # validation summaries
        self.model.eval()
        with torch.no_grad():
            Xv, yv, sv = [t.to(self.device) for t in self.val]
            logits = self.model(Xv)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            y_pred = logits.argmax(1).cpu().numpy()
            y_true = yv.cpu().numpy(); sensitive = sv.cpu().numpy()
            summ = summarize(y_true, probs, y_pred, sensitive)
            loss_val = float(cel(logits, yv).cpu().item())
            acc_val = float((y_pred==y_true).mean())

        return {
            "state_dict": self.get_state(),
            "val_loss": loss_val,
            "val_acc": acc_val,
            "summary": summ,
        }
