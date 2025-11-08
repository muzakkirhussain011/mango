#!/usr/bin/env python3
"""
Analyze and compare experimental results across algorithms.
Generates comparison tables and visualizations showing FairCare-FL's improvements.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_experiment_results(results_dir: Path) -> pd.DataFrame:
    """Load all experimental results into a DataFrame.

    Args:
        results_dir: Path to results directory

    Returns:
        DataFrame with all experimental results
    """
    results = []

    # Iterate through all algorithm directories
    for algo_dir in results_dir.iterdir():
        if not algo_dir.is_dir():
            continue

        algo_name = algo_dir.name

        # Iterate through all dataset directories
        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # Iterate through all seed directories
            for seed_dir in dataset_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                    continue

                seed = int(seed_dir.name.replace('seed', ''))

                # Load metrics from CSV if available
                metrics_file = seed_dir / 'metrics.csv'
                if metrics_file.exists():
                    try:
                        df = pd.read_csv(metrics_file)
                        # Get final (best) metrics
                        if len(df) > 0:
                            final_metrics = df.iloc[-1].to_dict()

                            # Extract test metrics
                            test_metrics = {k: v for k, v in final_metrics.items()
                                          if k.startswith('test/')}

                            result = {
                                'algorithm': algo_name,
                                'dataset': dataset_name,
                                'seed': seed,
                                **test_metrics
                            }
                            results.append(result)
                    except Exception as e:
                        print(f"Error loading {metrics_file}: {e}")

                # Also check for config to get experiment details
                config_file = seed_dir / 'config.yaml'
                if config_file.exists() and not metrics_file.exists():
                    # Try to load from experiment log or other sources
                    pass

    return pd.DataFrame(results)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each algorithm-dataset combination.

    Args:
        df: DataFrame with experimental results

    Returns:
        DataFrame with summary statistics
    """
    # Group by algorithm and dataset
    grouped = df.groupby(['algorithm', 'dataset'])

    # Compute mean and std for all numeric columns
    summary = grouped.agg(['mean', 'std'])

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary.reset_index()


def create_comparison_table(summary_df: pd.DataFrame,
                           metric_cols: List[str]) -> pd.DataFrame:
    """Create a formatted comparison table.

    Args:
        summary_df: DataFrame with summary statistics
        metric_cols: List of metrics to include

    Returns:
        Formatted comparison table
    """
    # Select relevant columns
    cols_to_keep = ['algorithm', 'dataset']
    for metric in metric_cols:
        cols_to_keep.extend([f'{metric}_mean', f'{metric}_std'])

    # Filter columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in summary_df.columns]

    table = summary_df[cols_to_keep].copy()

    # Format as mean ± std
    for metric in metric_cols:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'

        if mean_col in table.columns and std_col in table.columns:
            table[metric] = table.apply(
                lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}",
                axis=1
            )
            table = table.drop(columns=[mean_col, std_col])

    return table


def create_performance_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots comparing algorithms.

    Args:
        df: DataFrame with experimental results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define key metrics
    fairness_metrics = ['test/eo_gap', 'test/sp_gap', 'test/worst_group_f1']
    performance_metrics = ['test/accuracy', 'test/macro_f1', 'test/auroc']

    # 1. Accuracy vs Fairness Scatter Plot
    if 'test/accuracy' in df.columns and 'test/worst_group_f1' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            ax.scatter(
                algo_df['test/accuracy'],
                algo_df['test/worst_group_f1'],
                label=algo,
                alpha=0.6,
                s=100
            )

        ax.set_xlabel('Test Accuracy', fontsize=12)
        ax.set_ylabel('Worst-Group F1', fontsize=12)
        ax.set_title('Accuracy vs Fairness Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_vs_fairness.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Bar plots for each metric
    for metric in fairness_metrics + performance_metrics:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by algorithm and compute mean
        summary = df.groupby('algorithm')[metric].agg(['mean', 'std']).reset_index()

        # Create bar plot
        x = range(len(summary))
        ax.bar(x, summary['mean'], yerr=summary['std'],
               capsize=5, alpha=0.7, color='steelblue')

        ax.set_xticks(x)
        ax.set_xticklabels(summary['algorithm'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('test/', '').replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("test/", "").replace("_", " ").title()} Comparison',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best performer
        if 'gap' in metric:
            # Lower is better for gaps
            best_idx = summary['mean'].argmin()
        else:
            # Higher is better for other metrics
            best_idx = summary['mean'].argmax()

        ax.patches[best_idx].set_color('green')
        ax.patches[best_idx].set_alpha(0.8)

        plt.tight_layout()
        metric_name = metric.replace('test/', '').replace('/', '_')
        plt.savefig(output_dir / f'{metric_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}")


def compute_improvement_over_baseline(summary_df: pd.DataFrame,
                                      baseline: str = 'fedavg') -> pd.DataFrame:
    """Compute percentage improvement over baseline algorithm.

    Args:
        summary_df: DataFrame with summary statistics
        baseline: Name of baseline algorithm

    Returns:
        DataFrame with improvement percentages
    """
    improvements = []

    for dataset in summary_df['dataset'].unique():
        dataset_df = summary_df[summary_df['dataset'] == dataset]
        baseline_df = dataset_df[dataset_df['algorithm'] == baseline]

        if len(baseline_df) == 0:
            continue

        for _, row in dataset_df.iterrows():
            if row['algorithm'] == baseline:
                continue

            improvement = {
                'algorithm': row['algorithm'],
                'dataset': dataset
            }

            # Compute improvement for each metric
            for col in dataset_df.columns:
                if col.endswith('_mean') and col not in ['algorithm', 'dataset']:
                    metric_name = col.replace('_mean', '')
                    baseline_value = baseline_df[col].values[0]
                    current_value = row[col]

                    if baseline_value != 0:
                        if 'gap' in metric_name:
                            # For gap metrics, lower is better (negative improvement is good)
                            pct_improvement = ((baseline_value - current_value) / baseline_value) * 100
                        else:
                            # For other metrics, higher is better
                            pct_improvement = ((current_value - baseline_value) / baseline_value) * 100

                        improvement[metric_name] = pct_improvement

            improvements.append(improvement)

    return pd.DataFrame(improvements)


def main():
    """Main analysis function."""
    results_dir = Path('results/full_evaluation')

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("Loading experimental results...")
    df = load_experiment_results(results_dir)

    if len(df) == 0:
        print("No results found!")
        return

    print(f"Loaded {len(df)} experiments")
    print(f"Algorithms: {df['algorithm'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary_df = compute_summary_statistics(df)

    # Create comparison tables
    print("\nCreating comparison tables...")

    # Key metrics for comparison
    metrics_to_compare = [
        'test/accuracy',
        'test/macro_f1',
        'test/worst_group_f1',
        'test/eo_gap',
        'test/sp_gap'
    ]

    comparison_table = create_comparison_table(summary_df, metrics_to_compare)

    # Save to CSV
    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    comparison_table.to_csv(output_dir / 'algorithm_comparison.csv', index=False)
    print(f"Comparison table saved to {output_dir / 'algorithm_comparison.csv'}")

    # Print summary to console
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    print(comparison_table.to_string(index=False))

    # Compute improvement over FedAvg baseline
    if 'fedavg' in summary_df['algorithm'].unique():
        print("\n\nComputing improvements over FedAvg baseline...")
        improvement_df = compute_improvement_over_baseline(summary_df, baseline='fedavg')

        if len(improvement_df) > 0:
            improvement_df.to_csv(output_dir / 'improvements_over_fedavg.csv', index=False)
            print(f"\nImprovements saved to {output_dir / 'improvements_over_fedavg.csv'}")

            # Print FairCare-FL improvements
            faircare_improvements = improvement_df[
                improvement_df['algorithm'] == 'faircare_fl'
            ]

            if len(faircare_improvements) > 0:
                print("\n" + "="*80)
                print("FAIRCARE-FL IMPROVEMENTS OVER FEDAVG")
                print("="*80)
                print(faircare_improvements.to_string(index=False))

    # Create visualization plots
    print("\n\nCreating visualization plots...")
    create_performance_plots(df, output_dir / 'plots')

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
