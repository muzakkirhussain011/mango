from __future__ import annotations
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..config import ExperimentConfig
from .utils import set_seed, get_device
from ..data.partition import make_federated_splits
from ..models.classifier import make_classifier
from .server import Server
from .secure_agg import SecureAggregator
from ..algos import fedavg, fedprox, qffl, afl, fairfate, fairfed, fedgft, faircare_fl
from ..algos.aggregator import FairMomentumAggregator
from .client import Client

@dataclass
class TrainArtifacts:
    history: List[Dict[str, Any]]

def build_algo(name: str):
    return {
        "fedavg": fedavg,
        "fedprox": fedprox,
        "qffl": qffl,
        "afl": afl,
        "fairfate": fairfate,
        "fairfed": fairfed,
        "fedgft": fedgft,
        "faircare_fl": faircare_fl,
    }[name]

def run_experiment(config: ExperimentConfig) -> TrainArtifacts:
    set_seed(config.algorithm.seed)
    device = get_device(config.algorithm.device)
    (train_parts, val, input_dim, n_classes, sens_present) = make_federated_splits(
        dataset=config.data.name,
        n_clients=config.data.n_clients,
        alpha=config.data.dirichlet_alpha,
        sensitive=config.data.sensitive_feature,
        cache_dir=config.data.cache_dir,
    )

    clients = []
    train_loaders = []
    for cid, (x, y, a) in enumerate(train_parts):
        model = make_classifier(input_dim, n_classes).to(device)
        ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long),
                           torch.tensor(a, dtype=torch.long) if sens_present else torch.zeros(len(y), dtype=torch.long))
        train_loaders.append(DataLoader(ds, batch_size=config.algorithm.batch_size, shuffle=True, drop_last=False))
        clients.append(Client(
            cid=cid, model=model, device=device,
            lambda_g=config.algorithm.lambda_g,
            lambda_c=config.algorithm.lambda_c,
            fairness_mode=config.algorithm.fairness_mode,
            qffl_q=config.algorithm.qffl_q,
            fedprox_mu=config.algorithm.fedprox_mu,
        ))

    val_ds = TensorDataset(torch.tensor(val[0], dtype=torch.float32),
                           torch.tensor(val[1], dtype=torch.long),
                           torch.tensor(val[2], dtype=torch.long) if sens_present else torch.zeros(len(val[1]), dtype=torch.long))
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)

    base_model = make_classifier(input_dim, n_classes).to(device)
    algo_mod = build_algo(config.algorithm.name)

    aggregator = algo_mod.make_aggregator(sens_present=sens_present)
    momentum = FairMomentumAggregator(beta=config.algorithm.beta_momentum, model=base_model, device=device)
    server = Server(base_model, device, aggregator, momentum, SecureAggregator(),
                    target_gap=config.algorithm.target_fair_gap,
                    adapt_every=config.algorithm.adapt_every, adapt_scale=config.algorithm.adapt_scale)

    history = []
    for r in range(config.algorithm.max_rounds):
        out = server.run_round(clients, train_loaders, val_loader, sens_present, config.algorithm)
        if (r + 1) % config.log_every == 0:
            print(f"[Round {out['round']:03d}] {out['metrics']}")
        history.append(out)

    # Save history
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(config.output_dir)/"history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return TrainArtifacts(history=history)
