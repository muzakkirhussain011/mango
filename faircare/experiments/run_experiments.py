# faircare/experiments/run_experiments.py
import os, argparse, numpy as np, torch
from ..config import Defaults
from ..core.utils import set_seed, ensure_dir, save_json
from ..data import load_heart, load_adult, make_synth, dirichlet_partition, load_mimic_demo, load_eicu_subset
from ..models.classifier import MLPClassifier
from ..core.server import Server
from ..core.client import Client
from ..core.trainer import Trainer
from ..algos import FedAvg, FedProx, QFFL, AFL, FairFed, FedGFT, FairFATE, FairCareFL

def make_clients(X, y, s, num_clients, alpha, seed, input_dim):
    parts = dirichlet_partition(len(y), num_clients, alpha=alpha, seed=seed)
    clients = []
    for cid, idx in enumerate(parts):
        if len(idx) < 10:
            continue
        n = len(idx); ntr = int(0.8*n)
        idx_tr, idx_va = idx[:ntr], idx[ntr:]
        clients.append(Client(cid, X[idx_tr], y[idx_tr], s[idx_tr],
                                   X[idx_va], y[idx_va], s[idx_va],
                                   input_dim=input_dim))
    return clients

def get_data(name, sensitive):
    if name=="heart":  return load_heart()
    if name=="adult":  return load_adult(sensitive=sensitive)
    if name=="synth":  return make_synth()
    if name=="mimic":  return load_mimic_demo()
    if name=="eicu":   return load_eicu_subset()
    raise ValueError(f"Unknown dataset {name}")

def build_algo(args):
    if args.algo=="fedavg":   return FedAvg(args.local_epochs, args.batch_size, args.lr)
    if args.algo=="fedprox":  return FedProx(args.local_epochs, args.batch_size, args.lr, 0.01)
    if args.algo=="qffl":     return QFFL(args.local_epochs, args.batch_size, args.lr, args.q)
    if args.algo=="afl":      return AFL(args.local_epochs, args.batch_size, args.lr)
    if args.algo=="fairfed":  return FairFed(args.local_epochs, args.batch_size, args.lr)
    if args.algo=="fedgft":   return FedGFT(args.local_epochs, args.batch_size, args.lr, args.lambdaG)
    if args.algo=="fairfate": return FairFATE(args.local_epochs, args.batch_size, args.lr)
    if args.algo=="faircare": return FairCareFL(args.local_epochs, args.batch_size, args.lr, args.lambdaG, args.lambdaC, args.lambdaA, args.q, args.beta, use_adv=not args.no_adv)
    raise ValueError("Unknown algo")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="heart", choices=["heart","adult","synth","mimic","eicu"])
    p.add_argument("--algo", type=str, default="faircare")
    p.add_argument("--num_clients", type=int, default=Defaults.num_clients)
    p.add_argument("--rounds", type=int, default=Defaults.rounds)
    p.add_argument("--local_epochs", type=int, default=Defaults.local_epochs)
    p.add_argument("--batch_size", type=int, default=Defaults.batch_size)
    p.add_argument("--lr", type=float, default=Defaults.lr)
    p.add_argument("--beta", type=float, default=Defaults.beta_momentum)
    p.add_argument("--lambdaG", type=float, default=Defaults.lambdaG)
    p.add_argument("--lambdaC", type=float, default=Defaults.lambdaC)
    p.add_argument("--lambdaA", type=float, default=Defaults.lambdaA)
    p.add_argument("--q", type=float, default=Defaults.q)
    p.add_argument("--dirichlet_alpha", type=float, default=Defaults.dirichlet_alpha)
    p.add_argument("--sensitive_attr", type=str, default=Defaults.sensitive_attr)
    p.add_argument("--seed", type=int, default=Defaults.seed)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--no-adv", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    X, y, s, feature_names = get_data(args.dataset, args.sensitive_attr)
    input_dim = X.shape[1]
    clients = make_clients(X, y, s, args.num_clients, args.dirichlet_alpha, args.seed, input_dim)

    model = MLPClassifier(input_dim)
    server = Server(model, beta=args.beta)
    trainer = Trainer(server, clients, outdir=args.outdir, sensitive_attr=args.sensitive_attr)

    algo = build_algo(args)
    cfg_dump = vars(args)
    cfg_dump["n_clients_effective"] = len(clients)
    save_json(cfg_dump, os.path.join(args.outdir, "config.json"))

    hist = trainer.train(args.rounds, algo)
    save_json(hist, os.path.join(args.outdir, "history.json"))

if __name__ == "__main__":
    main()
