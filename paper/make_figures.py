# paper/make_figures.py
import os, argparse, csv
import matplotlib.pyplot as plt

def load_csv(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if k!="round" else int(v) for k,v in row.items()})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    metrics = load_csv(os.path.join(args.indir, "metrics.csv"))
    rounds = [m["round"] for m in metrics]
    acc = [m["global_accuracy"] for m in metrics]
    dp = [m["global_dp_gap"] for m in metrics]
    eo = [m["global_eo_gap"] for m in metrics]

    plt.figure()
    plt.plot(rounds, acc, marker="o")
    plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("Global Accuracy vs Round")
    plt.savefig(os.path.join(args.outdir, "accuracy_vs_round.png"), bbox_inches="tight")

    plt.figure()
    plt.plot(rounds, dp, marker="o", label="DP gap")
    plt.plot(rounds, eo, marker="s", label="EO gap")
    plt.xlabel("Round"); plt.ylabel("Gap"); plt.title("Fairness Gaps vs Round"); plt.legend()
    plt.savefig(os.path.join(args.outdir, "fairness_gaps_vs_round.png"), bbox_inches="tight")

if __name__ == "__main__":
    main()
