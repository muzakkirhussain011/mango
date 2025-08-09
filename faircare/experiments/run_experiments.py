# faircare/experiments/run_experiments.py
import os, json, argparse, random
import numpy as np
import torch
from faircare.data.heart import load_heart
from faircare.data.adult import load_adult
from faircare.data.partition import dirichlet_partition
from faircare.models.classifier import MLPClassifier
from faircare.models.adversary import LogitAdversary
from faircare.core.client import local_train
from faircare.core.evaluation import evaluate_union, evaluate_with_sweep
from faircare.algos.faircare_fl import FairCareFL

def get_data(name: str, sensitive: str):
    name = name.lower()
    if name == "heart":
        return load_heart()
    elif name == "adult":
        return load_adult()
    else:
        raise ValueError(f"Unknown dataset {name}")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="heart")
    ap.add_argument("--sensitive_attr", type=str, default="sex")
    ap.add_argument("--algo", type=str, default="faircare",
                    choices=["faircare","fedavg","fedprox","qffl","afl","fairfed","fedgft","fairfate"])
    ap.add_argument("--num_clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambdaG", type=float, default=2.0)
    ap.add_argument("--lambdaC", type=float, default=0.5)
    ap.add_argument("--lambdaA", type=float, default=0.5)
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.5)
    ap.add_argument("--server_lr", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pcgrad", action="store_true", help="Use PCGrad on clients")
    ap.add_argument("--global_eval", action="store_true", help="Evaluate on union each round")
    ap.add_argument("--threshold_sweep", action="store_true", help="Also compute best-EO/DP thresholds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    X, y, s, feat_names = get_data(args.dataset, args.sensitive_attr)

    # split into clients - FIX: pass len(y) instead of y
    parts = dirichlet_partition(len(y), num_clients=args.num_clients, alpha=args.dirichlet_alpha, seed=args.seed)
    clients = []
    for idxs in parts:
        clients.append((X[idxs], y[idxs], s[idxs]))

    # construct a small shared validation union for global evaluation (20%)
    n = len(y)
    perm = np.random.RandomState(args.seed).permutation(n)
    val_k = max(int(0.2 * n), 50)
    val_idx, train_idx = perm[:val_k], perm[val_k:]
    X_val, y_val, s_val = X[val_idx], y[val_idx], s[val_idx]

    # init model
    model = MLPClassifier(in_dim=X.shape[1], hidden=64, out_dim=2)
    adv = LogitAdversary(in_dim=2)

    # choose algo object (server)
    if args.algo == "faircare":
        server_algo = FairCareFL(
            model_params_template=[p.detach().cpu().numpy() for p in model.parameters()],
            q=args.q, gamma=1.0, temperature=args.temperature,
            server_lr=args.server_lr, beta1=args.beta, beta2=0.999, eps=1e-8,
            scaffold_c=0.0
        )
    elif args.algo == "fedavg":
        server_algo = FedAvg()
    elif args.algo == "fedprox":
        server_algo = FedProx(mu=0.01)
    elif args.algo == "qffl":
        server_algo = QFFL(q=args.q)
    elif args.algo == "afl":
        server_algo = AFL()
    elif args.algo == "fairfed":
        server_algo = FairFed()
    elif args.algo == "fedgft":
        server_algo = FedGFT(lambdaG=args.lambdaG)
    elif args.algo == "fairfate":
        server_algo = FairFATE(beta=args.beta)
    else:
        raise ValueError(args.algo)

    metrics_path = os.path.join(args.outdir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("round,global_accuracy,global_auroc,global_dp_gap,global_eo_gap\n")

    # federated rounds
    for r in range(1, args.rounds + 1):
        # broadcast current weights
        w0 = [p.detach().cpu().numpy().copy() for p in model.parameters()]

        client_reports, deltas = [], []
        for (Xi, yi, si) in clients:
            m_i = MLPClassifier(in_dim=X.shape[1], hidden=64, out_dim=2)
            m_i.load_state_dict(model.state_dict())
            opt = torch.optim.Adam(m_i.parameters(), lr=args.lr)
            rep, d = local_train(
                m_i, adv, opt, Xi, yi, si,
                batch_size=args.batch_size, local_epochs=args.local_epochs,
                lambdaG=args.lambdaG, lambdaA=args.lambdaA,
                use_pcgrad=args.pcgrad, device="cpu"
            )
            client_reports.append(rep)
            deltas.append(d["delta"])

        # server aggregation
        if args.algo == "faircare":
            # FairCareFL has its own aggregate method
            new_params = server_algo.aggregate(w0, deltas, client_reports)
        else:
            # Other algorithms use simple weighted averaging
            from faircare.algos import aggregator
            
            # Get weights based on algorithm
            if args.algo == "fedavg":
                weights = aggregator.weights_fedavg(client_reports)
            elif args.algo == "fedprox":
                weights = aggregator.weights_fedavg(client_reports)
            elif args.algo == "qffl":
                weights = aggregator.weights_qffl(client_reports, args.q)
            elif args.algo == "afl":
                weights = aggregator.weights_afl(client_reports, boost=3.0)
            elif args.algo == "fairfed":
                weights = aggregator.weights_fairfed(client_reports)
            elif args.algo == "fedgft":
                weights = aggregator.weights_fedavg(client_reports)
            elif args.algo == "fairfate":
                weights = aggregator.weights_fairfed(client_reports)
            else:
                weights = aggregator.weights_fedavg(client_reports)
            
            # Simple weighted average of deltas
            import torch
            new_params = []
            for i in range(len(w0)):
                weighted_delta = np.zeros_like(deltas[0][i])
                for w, delta in zip(weights, deltas):
                    weighted_delta += w * delta[i]
                new_params.append(w0[i] + weighted_delta)

        # update model
        with torch.no_grad():
            for p, npar in zip(model.parameters(), new_params):
                p.copy_(torch.tensor(npar, dtype=p.dtype))

        # global union eval
        if args.global_eval:
            res = evaluate_union(model, X_val, y_val, s_val, threshold=0.5)
            with open(metrics_path, "a") as f:
                f.write(f"{r},{res['accuracy']:.6f},{res['auroc']:.6f},{res['dp_gap']:.6f},{res['eo_gap']:.6f}\n")

    # optional sweep and summary JSON
    summary = {}
    if args.global_eval:
        summary["final"] = evaluate_union(model, X_val, y_val, s_val, threshold=0.5)
        if args.threshold_sweep:
            summary["sweep"] = evaluate_with_sweep(model, X_val, y_val, s_val)
    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
