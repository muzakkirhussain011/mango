"""Generate LaTeX tables for paper."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List


def load_results(results_dir: str) -> Dict:
    """Load aggregated results."""
    results_path = Path(results_dir) / "aggregated_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path) as f:
        return json.load(f)


def create_main_table(results: Dict) -> str:
    """Create main comparison table."""
    rows = []
    
    algorithms = ["fedavg", "fedprox", "qffl", "afl", "fairfate", "faircare_fl"]
    metrics = ["accuracy", "worst_group_F1", "EO_gap", "max_group_gap"]
    
    for algo in algorithms:
        if algo not in results:
            continue
        
        algo_results = results[algo]
        row = {"Algorithm": algo.upper().replace("_", "-")}
        
        for metric in metrics:
            values = [r.get(f"final_{metric}", r.get(metric, 0)) 
                     for r in algo_results]
            
            if values:
                mean = np.mean(values)
                std = np.std(values)
                
                # Format based on metric type
                if "gap" in metric.lower():
                    # Lower is better for gaps
                    row[metric] = f"{mean:.3f} ± {std:.3f}"
                else:
                    # Higher is better for accuracy/F1
                    row[metric] = f"{mean:.3f} ± {std:.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Highlight best results
    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1)
    )
    
    # Add caption and label
    latex = latex.replace("\\begin{tabular}", 
                         "\\begin{table}[h]\n\\centering\n\\begin{tabular}")
    latex = latex.replace("\\end{tabular}",
                         "\\end{tabular}\n\\caption{Comparison of algorithms on fairness metrics (mean ± std over 5 seeds)}\n\\label{tab:main_results}\n\\end{table}")
    
    return latex


def create_statistical_table(results: Dict) -> str:
    """Create statistical significance table."""
    from scipy import stats
    
    baseline = "fedavg"
    if baseline not in results:
        return ""
    
    baseline_results = results[baseline]
    rows = []
    
    for algo in results:
        if algo == baseline:
            continue
        
        algo_results = results[algo]
        
        # Compare worst_group_F1
        baseline_f1 = [r.get("final_worst_group_F1", 0) for r in baseline_results]
        algo_f1 = [r.get("final_worst_group_F1", 0) for r in algo_results]
        
        if len(baseline_f1) == len(algo_f1) and len(baseline_f1) > 1:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(algo_f1, baseline_f1)
            
            # Effect size (Cohen's d)
            diff = np.array(algo_f1) - np.array(baseline_f1)
            cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            row = {
                "Algorithm": algo.upper().replace("_", "-"),
                "Mean Diff": f"{np.mean(diff):.4f}",
                "Cohen's d": f"{cohen_d:.3f}",
                "p-value": f"{p_value:.4f}",
                "Significant": "✓" if p_value < 0.05 else ""
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc"
    )
    
    latex = latex.replace("\\begin{tabular}",
                         "\\begin{table}[h]\n\\centering\n\\begin{tabular}")
    latex = latex.replace("\\end{tabular}",
                         "\\end{tabular}\n\\caption{Statistical comparison with FedAvg baseline}\n\\label{tab:statistical}\n\\end{table}")
    
    return latex


def main():
    """Generate all tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/tables")
    parser.add_argument("--output_dir", default="paper/tables")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    # Generate tables
    main_table = create_main_table(results)
    stat_table = create_statistical_table(results)
    
    # Save tables
    with open(output_dir / "main_results.tex", "w") as f:
        f.write(main_table)
    
    with open(output_dir / "statistical.tex", "w") as f:
        f.write(stat_table)
    
    print(f"Tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
