# paper/tables.py
import json, os, numpy as np, scipy.stats as st

def summarize_runs(result_paths, keys=("accuracy","EO_gap","SP_gap")):
    rows = {}
    for name, path in result_paths.items():
        with open(path) as f:
            X = json.load(f)  # list of dicts per seed
        out = {}
        for k in keys:
            arr = np.array([r[k] for r in X])
            m = arr.mean()
            ci = st.t.interval(0.95, len(arr)-1, loc=m, scale=st.sem(arr)) if len(arr)>1 else (m,m)
            out[k] = {"mean": float(m), "ci95": [float(ci[0]), float(ci[1])]}
        rows[name] = out
    return rows

def effect_size(a, b):
    # Cohen's d: positive when a is better (higher accuracy, lower gap is "negative-better" though)
    a_m, b_m = np.mean(a), np.mean(b)
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1))/2.0)
    return float((a_m - b_m)/max(1e-12, s))

def main():
    # Example input mapping (fill with your produced JSON from scripts/reproduce_tables.sh)
    result_paths = {
        "FedAvg": "paper/tables/fedavg_adult.json",
        "qFFL": "paper/tables/qffl_adult.json",
        "FairFed": "paper/tables/fairfed_adult.json",
        "FAIR-FATE": "paper/tables/fairfate_adult.json",
        "FedGFT": "paper/tables/fedgft_adult.json",
        "FairCare-F1": "paper/tables/faircare_f1_adult.json",
    }
    rows = summarize_runs(result_paths)
    # Pretty print
    import pprint
    pprint.pprint(rows)

if __name__ == "__main__":
    main()
