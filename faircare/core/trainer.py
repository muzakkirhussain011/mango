# faircare/core/trainer.py
from typing import Dict, Any, Tuple, List
import torch
from .server import Server
from .utils import make_model
from ..data.partition import make_federated_splits

def run_experiment(cfg: Dict[str, Any]):
    seed = cfg.get("seed", 13)
    torch.manual_seed(seed)

    # data
    (train_parts, val, input_dim, n_classes, sens_present) = make_federated_splits(
        dataset=cfg["dataset"]["name"],
        cache_dir=cfg["dataset"].get("cache_dir", "./_cache"),
        sensitive=cfg["dataset"].get("sensitive", "sex"),
        n_clients=cfg["dataset"].get("n_clients", 10),
        iid_frac=cfg["dataset"].get("iid_frac", 0.0)
    )

    # models
    model_ctor = lambda: make_model(input_dim)
    clients = []
    from .client import Client
    for _ in range(len(train_parts)):
        clients.append(Client(model_ctor=model_ctor, adversary_ctor=None, device=cfg.get("device","cpu")))

    server = Server(model_ctor=model_ctor, device=cfg.get("device","cpu"))

    # loaders
    def make_loader(split, batch=128, shuffle=True):
        from torch.utils.data import DataLoader, TensorDataset
        x, y, a = split["x"], split["y"], split["a"]
        ds = TensorDataset(x, y, a)
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    train_loaders = [make_loader(s) for s in train_parts]
    val_loader = make_loader(val, batch=1024, shuffle=False)

    rounds = cfg.get("rounds", 20)
    for r in range(1, rounds + 1):
        out = server.run_round(clients, train_loaders, val_loader, sens_present, cfg["algorithm"])
        msg = {k: round(float(v), 6) for k, v in out.items()}
        print(f"[Round {r:03d}] {msg}")
