# faircare/experiments/run_sweep.py
import os, argparse, csv, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="heart")
    ap.add_argument("--algos", nargs="+", default=["faircare","fedgft","qffl","afl"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,43,44,45,46])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    index_csv = os.path.join(args.outdir, "index.csv")
    with open(index_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["algo","seed","path"])
        for algo in args.algos:
            for seed in args.seeds:
                run_dir = os.path.join(args.outdir, f"{algo}_seed{seed}")
                cmd = [
                    sys.executable, "-m", "faircare.experiments.run_experiments",
                    "--dataset", args.dataset, "--algo", algo,
                    "--num_clients", "5", "--rounds", str(args.rounds),
                    "--local_epochs", "1", "--batch_size", "64",
                    "--lr", "1e-3", "--lambdaG", "2.0", "--lambdaC", "0.5", "--lambdaA", "0.5",
                    "--q", "0.5", "--beta", "0.9", "--dirichlet_alpha", "0.5",
                    "--sensitive_attr", "sex", "--global_eval",
                    "--seed", str(seed), "--outdir", run_dir
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                w.writerow([algo, seed, run_dir])
    print("Wrote", index_csv)

if __name__ == "__main__":
    main()
