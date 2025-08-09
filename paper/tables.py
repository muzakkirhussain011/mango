# paper/tables.py
import os, argparse, csv, glob
from scipy import stats
from tabulate import tabulate

def read_last(indir):
    with open(os.path.join(indir, "metrics.csv"), "r") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]

def collect_runs(index_csv):
    out = {}
    with open(index_csv, "r") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        algo = r["algo"]; path = r["path"]
        last = read_last(path)
        out.setdefault(algo, []).append(last)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="either a single run dir or a sweep dir containing index.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if os.path.exists(os.path.join(args.indir, "index.csv")):
        results = collect_runs(os.path.join(args.indir, "index.csv"))
        # compare faircare vs best baseline (by mean EO)
        means = {algo: sum(float(x["global_eo_gap"]) for x in xs)/len(xs) for algo,xs in results.items()}
        baseline = sorted((a for a in results if a!="faircare"), key=lambda a: means[a])[0]
        fair_vals = [float(x["global_eo_gap"]) for x in results["faircare"]]
        base_vals = [float(x["global_eo_gap"]) for x in results[baseline]]
        t,p = stats.ttest_rel(base_vals, fair_vals)
        # LaTeX table
        lines = [["Algo","Accuracy","DP","EO","n"]]
        for algo, xs in results.items():
            acc = sum(float(x["global_accuracy"]) for x in xs)/len(xs)
            dp  = sum(float(x["global_dp_gap"]) for x in xs)/len(xs)
            eo  = sum(float(x["global_eo_gap"]) for x in xs)/len(xs)
            lines.append([algo, f"{acc:.3f}", f"{dp:.3f}", f"{eo:.3f}", len(xs)])
        tex = "\\begin{tabular}{lcccc}\n\\hline\nAlgo & Acc & DP & EO & n \\\\\n\\hline\n"
        for row in lines[1:]:
            tex += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n"
        tex += "\\hline\n\\end{tabular}\n"
        with open(os.path.join(args.outdir, "results_table.tex"), "w") as f:
            f.write(tex)
        with open("paper/RESULTS.md", "w") as f:
            f.write("# Results Summary\n\n")
            f.write(f"- Baseline compared: **{baseline}**\n")
            f.write(f"- Paired t-test EO gap (baseline vs faircare): t={t:.3f}, p={p:.4f}\n")
            f.write("\n## Means\n")
            f.write(tabulate(lines[1:], headers=lines[0], tablefmt="github"))
    else:
        last = read_last(args.indir)
        tex = r"""\begin{tabular}{lcccc}
\hline
Round & Accuracy & AUROC & DP Gap & EO Gap \\
\hline
%s & %.3f & %.3f & %.3f & %.3f \\
\hline
\end{tabular}
""" % (last["round"], float(last["global_accuracy"]), float(last["global_auroc"]), float(last["global_dp_gap"]), float(last["global_eo_gap"]))
        with open(os.path.join(args.outdir, "results_table.tex"), "w") as f:
            f.write(tex)
        with open("paper/RESULTS.md", "w") as f:
            f.write(f"# Results Summary\n\n")
            f.write(f"- Final Accuracy: {float(last['global_accuracy']):.3f}\n")
            f.write(f"- Final DP Gap: {float(last['global_dp_gap']):.3f}\n")
            f.write(f"- Final EO Gap: {float(last['global_eo_gap']):.3f}\n")

if __name__ == "__main__":
    main()
