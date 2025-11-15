#!/usr/bin/env python3
"""
Comprehensive Analysis of ALL Experimental Results
Compares FairCare-FL against ALL baselines across ALL datasets
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def find_latest_result(base_path):
    """Find the most recent result directory."""
    if not os.path.exists(base_path):
        return None

    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        return None

    subdirs.sort(reverse=True)
    latest_dir = os.path.join(base_path, subdirs[0])
    result_file = os.path.join(latest_dir, 'final_results.json')

    if os.path.exists(result_file):
        return result_file
    return None

def load_all_results(results_dir):
    """Load all experimental results."""
    algorithms = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed', 'faircare_fl']
    datasets = ['adult', 'compas', 'mimic', 'eicu']
    seeds = [0, 1, 2]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset in datasets:
        for algo in algorithms:
            for seed in seeds:
                base_path = os.path.join(results_dir, dataset, algo, f'seed{seed}')
                result_file = find_latest_result(base_path)

                if result_file:
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            metrics = data.get('final_metrics', data)
                            results[dataset][algo][seed] = {
                                'accuracy': metrics.get('test/accuracy', 0.0),
                                'worst_group_f1': metrics.get('test/worst_group_f1', 0.0),
                                'eo_gap': metrics.get('test/eo_gap', 0.0),
                                'fpr_gap': metrics.get('test/fpr_gap', 0.0),
                                'sp_gap': metrics.get('test/sp_gap', 0.0),
                                'macro_f1': metrics.get('test/macro_f1', 0.0),
                                'auroc': metrics.get('test/auroc', 0.5)
                            }
                            print(f"[OK] Loaded: {dataset}/{algo}/seed{seed}")
                    except Exception as e:
                        print(f"[ERROR] {dataset}/{algo}/seed{seed}: {e}")
                else:
                    print(f"[MISSING] {dataset}/{algo}/seed{seed}")

    return results

def compute_statistics(results):
    """Compute mean and std for each algorithm-dataset pair."""
    stats = defaultdict(lambda: defaultdict(dict))

    for dataset, algos in results.items():
        for algo, seed_results in algos.items():
            if not seed_results:
                continue

            metrics = defaultdict(list)
            for seed, result in seed_results.items():
                for key, value in result.items():
                    metrics[key].append(value)

            stats[dataset][algo] = {
                key: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for key, values in metrics.items() if values
            }

    return stats

def print_dataset_comparison(stats, dataset):
    """Print comparison table for a specific dataset."""
    print(f"\n{'='*120}")
    print(f"DATASET: {dataset.upper()}")
    print('='*120)

    algorithms = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed', 'faircare_fl']

    # Header
    print(f"\n{'Algorithm':<15} {'Accuracy':<20} {'Worst-Group F1':<20} {'EO Gap':<20} {'Macro F1':<20}")
    print("-" * 120)

    # Data rows
    for algo in algorithms:
        if algo in stats[dataset] and stats[dataset][algo]:
            s = stats[dataset][algo]
            acc = s.get('accuracy', {'mean': 0, 'std': 0})
            wg_f1 = s.get('worst_group_f1', {'mean': 0, 'std': 0})
            eo = s.get('eo_gap', {'mean': 0, 'std': 0})
            macro = s.get('macro_f1', {'mean': 0, 'std': 0})

            print(f"{algo:<15} "
                  f"{acc['mean']:.4f}±{acc['std']:.4f}  "
                  f"{wg_f1['mean']:.4f}±{wg_f1['std']:.4f}  "
                  f"{eo['mean']:.4f}±{eo['std']:.4f}  "
                  f"{macro['mean']:.4f}±{macro['std']:.4f}")
        else:
            print(f"{algo:<15} {'[NO DATA]':<20}")

    print('='*120)

def compare_faircare_to_baselines_per_dataset(stats, dataset):
    """Compare FairCare-FL to best baseline for a specific dataset."""
    print(f"\n{'='*100}")
    print(f"FAIRCARE-FL vs BEST BASELINE: {dataset.upper()}")
    print('='*100)

    baseline_algos = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed']

    # Check if we have data
    if dataset not in stats or 'faircare_fl' not in stats[dataset]:
        print(f"\n[ERROR] No FairCare-FL results for {dataset}")
        return

    # Find best baseline for each metric
    available_baselines = [a for a in baseline_algos if a in stats[dataset] and stats[dataset][a]]

    if not available_baselines:
        print(f"\n[ERROR] No baseline results for {dataset}")
        return

    best_acc = max([stats[dataset][a]['accuracy']['mean'] for a in available_baselines])
    best_wg_f1 = max([stats[dataset][a]['worst_group_f1']['mean'] for a in available_baselines])
    min_eo_gap = min([stats[dataset][a]['eo_gap']['mean'] for a in available_baselines])

    # FairCare-FL metrics
    fc_stats = stats[dataset]['faircare_fl']
    fc_acc = fc_stats['accuracy']['mean']
    fc_wg_f1 = fc_stats['worst_group_f1']['mean']
    fc_eo_gap = fc_stats['eo_gap']['mean']

    print(f"\nMetric              | Best Baseline | FairCare-FL  | Delta      | Assessment")
    print("-" * 90)

    # Accuracy
    acc_delta = fc_acc - best_acc
    acc_pct = (acc_delta / best_acc) * 100 if best_acc > 0 else 0
    acc_status = "[PASS]" if acc_delta > -0.01 else "[FAIL]"
    print(f"Accuracy            | {best_acc:.4f}       | {fc_acc:.4f}     | {acc_delta:+.4f}   | {acc_status} ({acc_pct:+.1f}%)")

    # Worst-Group F1
    wg_delta = fc_wg_f1 - best_wg_f1
    wg_pct = (wg_delta / best_wg_f1) * 100 if best_wg_f1 > 0 else 0
    wg_status = "[PASS]" if wg_delta >= 0.03 or wg_pct >= 8.0 else "[FAIL]"
    print(f"Worst-Group F1      | {best_wg_f1:.4f}       | {fc_wg_f1:.4f}     | {wg_delta:+.4f}   | {wg_status} ({wg_pct:+.1f}%)")

    # EO Gap
    eo_delta = fc_eo_gap - min_eo_gap
    eo_reduction = ((min_eo_gap - fc_eo_gap) / min_eo_gap) * 100 if min_eo_gap > 0 else 0
    eo_status = "[PASS]" if eo_reduction >= 30.0 else "[FAIL]"
    print(f"EO Gap              | {min_eo_gap:.4f}       | {fc_eo_gap:.4f}     | {eo_delta:+.4f}   | {eo_status} ({eo_reduction:.1f}% reduction)")

    print('='*100)

def generate_summary_table(stats):
    """Generate overall summary table."""
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY: FairCare-FL Performance Across All Datasets")
    print('='*100)

    datasets = ['adult', 'compas', 'mimic', 'eicu']

    print(f"\n{'Dataset':<15} {'Acc (FC)':<15} {'Acc (Best)':<15} {'WG-F1 (FC)':<15} {'WG-F1 (Best)':<15} {'Status':<20}")
    print("-" * 100)

    for dataset in datasets:
        if dataset not in stats or 'faircare_fl' not in stats[dataset]:
            print(f"{dataset:<15} {'[NO DATA]':<15}")
            continue

        baseline_algos = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed']
        available_baselines = [a for a in baseline_algos if a in stats[dataset] and stats[dataset][a]]

        if not available_baselines:
            print(f"{dataset:<15} {'[NO BASELINES]':<15}")
            continue

        fc_acc = stats[dataset]['faircare_fl']['accuracy']['mean']
        fc_wg = stats[dataset]['faircare_fl']['worst_group_f1']['mean']
        best_acc = max([stats[dataset][a]['accuracy']['mean'] for a in available_baselines])
        best_wg = max([stats[dataset][a]['worst_group_f1']['mean'] for a in available_baselines])

        wg_improvement = ((fc_wg - best_wg) / best_wg * 100) if best_wg > 0 else 0
        acc_regression = ((best_acc - fc_acc) / best_acc * 100) if best_acc > 0 else 0

        status = "[PASS]" if wg_improvement >= 8.0 and acc_regression < 1.0 else "[FAIL]"

        print(f"{dataset:<15} {fc_acc:<15.4f} {best_acc:<15.4f} {fc_wg:<15.4f} {best_wg:<15.4f} {status:<20}")

    print('='*100)

def save_results_csv(stats, output_file):
    """Save results to CSV for further analysis."""
    rows = []
    for dataset, algos in stats.items():
        for algo, metrics in algos.items():
            row = {'dataset': dataset, 'algorithm': algo}
            for metric, values in metrics.items():
                row[f'{metric}_mean'] = values['mean']
                row[f'{metric}_std'] = values['std']
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n[SAVED] Results saved to: {output_file}")

def main():
    results_dir = 'results/full_evaluation'

    if not os.path.exists(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        print("        Run experiments first: bash scripts/run_comprehensive_evaluation.sh")
        sys.exit(1)

    print("="*120)
    print("COMPREHENSIVE ANALYSIS: All Algorithms Across All Datasets")
    print("="*120)

    # Load results
    print("\nLoading results...")
    results = load_all_results(results_dir)

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)

    # Print comparison tables for each dataset
    for dataset in ['adult', 'compas', 'mimic', 'eicu']:
        if dataset in stats and stats[dataset]:
            print_dataset_comparison(stats, dataset)
            compare_faircare_to_baselines_per_dataset(stats, dataset)

    # Generate overall summary
    generate_summary_table(stats)

    # Save to CSV
    save_results_csv(stats, f'{results_dir}/comprehensive_analysis.csv')

    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)
    print("\nNext steps:")
    print("  1. Review comparison tables above")
    print("  2. Check comprehensive_analysis.csv for detailed metrics")
    print("  3. Assess if FairCare-FL meets success criteria across all datasets")
    print("")

if __name__ == '__main__':
    main()
