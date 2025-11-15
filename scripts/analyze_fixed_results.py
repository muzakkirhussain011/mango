#!/usr/bin/env python3
"""
Analyze results from fixed baseline experiments.
Compare with previous buggy results to see the impact of the fix.
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

def load_results(results_dir):
    """Load all experimental results."""
    algorithms = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed', 'faircare_fl']
    datasets = ['adult']
    seeds = [0, 1, 2]

    results = defaultdict(lambda: defaultdict(list))

    for algo in algorithms:
        for dataset in datasets:
            for seed in seeds:
                base_path = os.path.join(results_dir, algo, dataset, f'seed{seed}')
                result_file = find_latest_result(base_path)

                if result_file:
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            metrics = data.get('final_metrics', data)
                            results[algo][dataset].append({
                                'seed': seed,
                                'accuracy': metrics.get('test/accuracy', 0.0),
                                'worst_group_f1': metrics.get('test/worst_group_f1', 0.0),
                                'eo_gap': metrics.get('test/eo_gap', 0.0),
                                'fpr_gap': metrics.get('test/fpr_gap', 0.0),
                                'sp_gap': metrics.get('test/sp_gap', 0.0),
                                'macro_f1': metrics.get('test/macro_f1', 0.0),
                                'auroc': metrics.get('test/auroc', 0.5)
                            })
                            print(f"[OK] Loaded: {algo}/{dataset}/seed{seed}")
                    except Exception as e:
                        print(f"[ERROR] {algo}/{dataset}/seed{seed}: {e}")
                else:
                    print(f"[MISSING] {algo}/{dataset}/seed{seed}")

    return results

def compute_statistics(results):
    """Compute mean and std for each algorithm-dataset pair."""
    stats = {}

    for algo, datasets in results.items():
        stats[algo] = {}
        for dataset, seed_results in datasets.items():
            if not seed_results:
                continue

            metrics = defaultdict(list)
            for result in seed_results:
                for key, value in result.items():
                    if key != 'seed':
                        metrics[key].append(value)

            stats[algo][dataset] = {
                key: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for key, values in metrics.items()
            }

    return stats

def print_comparison_table(stats):
    """Print comparison table."""
    print("\n" + "="*100)
    print("FIXED BASELINES: Performance Comparison")
    print("="*100)

    algorithms = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed', 'faircare_fl']
    dataset = 'adult'

    # Header
    print(f"\n{'Algorithm':<15} {'Accuracy':<20} {'Worst-Group F1':<20} {'EO Gap':<20} {'Macro F1':<20}")
    print("-" * 100)

    # Data rows
    for algo in algorithms:
        if algo in stats and dataset in stats[algo]:
            s = stats[algo][dataset]
            acc = s['accuracy']
            wg_f1 = s['worst_group_f1']
            eo = s['eo_gap']
            macro = s['macro_f1']

            print(f"{algo:<15} "
                  f"{acc['mean']:.4f}±{acc['std']:.4f}  "
                  f"{wg_f1['mean']:.4f}±{wg_f1['std']:.4f}  "
                  f"{eo['mean']:.4f}±{eo['std']:.4f}  "
                  f"{macro['mean']:.4f}±{macro['std']:.4f}")

    print("="*100)

def analyze_variance(stats):
    """Analyze variance across baselines."""
    print("\n" + "="*100)
    print("VARIANCE ANALYSIS: Are Baselines Different Now?")
    print("="*100)

    algorithms = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed']
    dataset = 'adult'

    # Collect means
    accuracies = []
    wg_f1s = []

    for algo in algorithms:
        if algo in stats and dataset in stats[algo]:
            accuracies.append(stats[algo][dataset]['accuracy']['mean'])
            wg_f1s.append(stats[algo][dataset]['worst_group_f1']['mean'])

    if len(accuracies) >= 2:
        acc_range = max(accuracies) - min(accuracies)
        acc_std = np.std(accuracies)
        wg_range = max(wg_f1s) - min(wg_f1s)
        wg_std = np.std(wg_f1s)

        print(f"\nAccuracy:")
        print(f"  Range: {min(accuracies):.4f} - {max(accuracies):.4f} (span: {acc_range:.4f})")
        print(f"  Std Dev: {acc_std:.4f}")

        print(f"\nWorst-Group F1:")
        print(f"  Range: {min(wg_f1s):.4f} - {max(wg_f1s):.4f} (span: {wg_range:.4f})")
        print(f"  Std Dev: {wg_std:.4f}")

        # Assessment
        print(f"\nAssessment:")
        if acc_std < 0.001 and wg_std < 0.001:
            print("  [WARNING] Baselines still nearly identical!")
            print("            Fix may not be working correctly.")
        elif acc_std < 0.01 and wg_std < 0.01:
            print("  [CAUTION] Baselines show low variance.")
            print("            May need more rounds or different hyperparameters.")
        else:
            print("  [SUCCESS] Baselines show meaningful variance!")
            print("            Fix is working correctly.")

    print("="*100)

def compare_faircare_to_baselines(stats):
    """Compare FairCare-FL to best baseline."""
    print("\n" + "="*100)
    print("FAIRCARE-FL vs BEST BASELINE")
    print("="*100)

    baseline_algos = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed']
    dataset = 'adult'

    # Find best baseline for each metric
    best_acc = max([stats[a][dataset]['accuracy']['mean']
                    for a in baseline_algos if a in stats and dataset in stats[a]])
    best_wg_f1 = max([stats[a][dataset]['worst_group_f1']['mean']
                      for a in baseline_algos if a in stats and dataset in stats[a]])
    min_eo_gap = min([stats[a][dataset]['eo_gap']['mean']
                      for a in baseline_algos if a in stats and dataset in stats[a]])

    # FairCare-FL metrics
    if 'faircare_fl' in stats and dataset in stats['faircare_fl']:
        fc_stats = stats['faircare_fl'][dataset]
        fc_acc = fc_stats['accuracy']['mean']
        fc_wg_f1 = fc_stats['worst_group_f1']['mean']
        fc_eo_gap = fc_stats['eo_gap']['mean']

        print(f"\nMetric              | Best Baseline | FairCare-FL  | Delta      | Assessment")
        print("-" * 90)

        # Accuracy
        acc_delta = fc_acc - best_acc
        acc_pct = (acc_delta / best_acc) * 100
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

        print("\n" + "="*90)
        print("SUCCESS CRITERIA (from requirements):")
        print("  - Worst-Group F1: ≥ +3% absolute OR ≥ +8% relative")
        print("  - EO/SP gaps: ≥ 30% reduction vs best baseline")
        print("  - Accuracy: < 1% regression")
        print("="*90)
    else:
        print("\n[ERROR] FairCare-FL results not found!")

    print("="*100)

def save_results_csv(stats, output_file):
    """Save results to CSV for further analysis."""
    rows = []
    for algo, datasets in stats.items():
        for dataset, metrics in datasets.items():
            row = {'algorithm': algo, 'dataset': dataset}
            for metric, values in metrics.items():
                row[f'{metric}_mean'] = values['mean']
                row[f'{metric}_std'] = values['std']
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n[SAVED] Results saved to: {output_file}")

def main():
    results_dir = 'results/fixed_baselines_evaluation'

    if not os.path.exists(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        print("        Run experiments first: bash scripts/run_fixed_baselines.sh")
        sys.exit(1)

    print("="*100)
    print("ANALYZING FIXED BASELINE RESULTS")
    print("="*100)

    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)

    # Print comparison table
    print_comparison_table(stats)

    # Analyze variance
    analyze_variance(stats)

    # Compare FairCare-FL to baselines
    compare_faircare_to_baselines(stats)

    # Save to CSV
    save_results_csv(stats, f'{results_dir}/analysis_summary.csv')

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("\nNext steps:")
    print("  1. Review the comparison tables above")
    print("  2. If baselines are now different: SUCCESS! Proceed with Phase 2 enhancements")
    print("  3. If baselines still identical: Investigate further")
    print("  4. Assess if FairCare-FL meets success criteria")
    print("")

if __name__ == '__main__':
    main()
