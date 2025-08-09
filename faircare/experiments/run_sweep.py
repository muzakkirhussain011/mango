# faircare/experiments/run_sweep.py
import os, argparse, itertools, subprocess, json, csv
from pathlib import Path

def run(args_list):
    print("RUN:", " ".join(args_list))
    out = subprocess.run(args_list, check=True, capture_output=True, text=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--algos", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    index_path = Path(args.outdir) / "index.csv"
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo","seed","path"])
        w.writeheader()

    for algo, seed in itertools.product(args.algos, args.seeds):
        run_dir = Path(args.outdir) / f"{algo}_seed{seed}"
        cmd = [
            "python","-m","faircare.experiments.run_experiments",
            "--dataset", args.dataset, "--algo", algo,
            "--num_clients","5","--rounds",str(args.rounds),
            "--outdir", str(run_dir),
            "--seed", str(seed),
        ]
        run(cmd)
        with open(index_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["algo","seed","path"])
            w.writerow({"algo":algo,"seed":seed,"path":str(run_dir)})

if __name__ == "__main__":
    main()
