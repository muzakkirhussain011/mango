from typing import Dict, List, Optional, Tuple, Any, Union, Protocol

"""Generate figures for paper."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def plot_convergence(results_dir: str, output_dir: str):
    """Plot convergence curves."""
    # Load metrics history
    metrics_files = Path(results_dir).glob("*/*/metrics.jsonl")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    algo_histories = {}
    
    for metrics_file in metrics_files:
        # Extract algorithm name
        parts = metrics_file.parts
        algo = parts[-3] if len(parts) >= 3 else "unknown"
        
        if algo not in algo_histories:
            algo_histories[algo] = []
        
        # Load metrics
        history = []
        with open(metrics_file) as f:
            for line in f:
                history.append(json.loads(line))
        
        algo_histories[algo].append(history)
    
    # Plot for each algorithm
    for algo, histories in algo_histories.items():
        if not histories:
            continue
        
        # Average across seeds
        max_rounds = max(len(h) for h in histories)
        
        avg_acc = []
        avg_eo_gap = []
        avg_worst_f1 = []
        avg_loss = []
        
        for round_idx in range(max_rounds):
            round_accs = []
            round_gaps = []
            round_f1s = []
            round_losses = []
            
            for history in histories:
                if round_idx < len(history):
                    round_accs.append(history[round_idx].get("val_accuracy", 0))
                    round_gaps.append(history[round_idx].get("val_EO_gap", 0))
                    round_f1s.append(history[round_idx].get("val_worst_group_F1", 0))
                    round_losses.append(history[round_idx].get("avg_train_loss", 0))
            
            if round_accs:
                avg_acc.append(np.mean(round_accs))
                avg_eo_gap.append(np.mean(round_gaps))
                avg_worst_f1.append(np.mean(round_f1s))
                avg_loss.append(np.mean(round_losses))
        
        rounds = range(1, len(avg_acc) + 1)
        
        # Plot
        axes[0, 0].plot(rounds, avg_acc, label=algo.upper())
        axes[0, 1].plot(rounds, avg_eo_gap, label=algo.upper())
        axes[1, 0].plot(rounds, avg_worst_f1, label=algo.upper())
        axes[1, 1].plot(rounds, avg_loss, label=algo.upper())
    
    # Configure subplots
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Validation Accuracy")
    axes[0, 0].set_title("Model Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("EO Gap")
    axes[0, 1].set_title("Equal Opportunity Gap")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Worst Group F1")
    axes[1, 0].set_title("Worst Group Performance")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Training Loss")
    axes[1, 1].set_title("Training Progress")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "convergence.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved to: {output_path}")


def plot_fairness_comparison(results_dir: str, output_dir: str):
    """Plot fairness metrics comparison."""
    # Load aggregated results
    results_path = Path(results_dir) / "aggregated_results.json"
    
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Prepare data for plotting
    data = []
    
    for algo, algo_results in results.items():
        for run in algo_results:
            data.append({
                "Algorithm": algo.upper().replace("_", "-"),
                "Worst Group F1": run.get("final_worst_group_F1", 0),
                "EO Gap": run.get("final_EO_gap", 0),
                "Max Gap": run.get("final_max_group_gap", 0)
            })
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plots for each metric
    sns.boxplot(data=df, x="Algorithm", y="Worst Group F1", ax=axes[0])
    axes[0].set_title("Worst Group F1 Score")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    sns.boxplot(data=df, x="Algorithm", y="EO Gap", ax=axes[1])
    axes[1].set_title("Equal Opportunity Gap")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    sns.boxplot(data=df, x="Algorithm", y="Max Gap", ax=axes[2])
    axes[2].set_title("Maximum Fairness Gap")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "fairness_comparison.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Fairness comparison saved to: {output_path}")


def main():
    """Generate all figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/full_evaluation")
    parser.add_argument("--output_dir", default="paper/figures")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    plot_convergence(args.results_dir, args.output_dir)
    plot_fairness_comparison(args.results_dir, args.output_dir)
    
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
