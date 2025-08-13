"""Summary and visualization utilities."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json
from pathlib import Path


def create_summary_table(
    results: Dict[str, List[Dict]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Create summary table from results.
    
    Args:
        results: Dictionary mapping algorithm names to lists of run results
        metrics: Metrics to include in table
    
    Returns:
        Summary DataFrame
    """
    if metrics is None:
        metrics = [
            "accuracy", "macro_F1", "worst_group_F1",
            "EO_gap", "FPR_gap", "SP_gap", "max_group_gap"
        ]
    
    summary_data = []
    
    for algo_name, runs in results.items():
        algo_summary = {"algorithm": algo_name}
        
        for metric in metrics:
            values = [run.get(metric, 0) for run in runs]
            if values:
                algo_summary[f"{metric}_mean"] = np.mean(values)
                algo_summary[f"{metric}_std"] = np.std(values)
        
        summary_data.append(algo_summary)
    
    return pd.DataFrame(summary_data)


def plot_fairness_metrics(
    history: Dict[str, List],
    save_path: Optional[str] = None
) -> None:
    """Plot fairness metrics over rounds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    rounds = range(1, len(history.get("accuracy", [])) + 1)
    
    # Accuracy
    if "accuracy" in history:
        axes[0, 0].plot(rounds, history["accuracy"], label="Accuracy")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Fairness gaps
    gap_metrics = ["EO_gap", "FPR_gap", "SP_gap"]
    for metric in gap_metrics:
        if metric in history:
            axes[0, 1].plot(rounds, history[metric], label=metric)
    
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Gap")
    axes[0, 1].set_title("Fairness Gaps")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Group-specific metrics
    if "g0_TPR" in history and "g1_TPR" in history:
        axes[1, 0].plot(rounds, history["g0_TPR"], label="Group 0 TPR")
        axes[1, 0].plot(rounds, history["g1_TPR"], label="Group 1 TPR")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("Group TPR Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss
    if "train_loss" in history:
        axes[1, 1].plot(rounds, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[1, 1].plot(rounds, history["val_loss"], label="Val Loss")
    
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Training Progress")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def export_results_json(
    results: Dict,
    output_path: str
) -> None:
    """Export results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def export_latex_table(
    df: pd.DataFrame,
    output_path: str,
    caption: str = "Experimental Results",
    label: str = "tab:results"
) -> None:
    """Export DataFrame to LaTeX table."""
    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(df.columns) - 1)
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(latex_str)
