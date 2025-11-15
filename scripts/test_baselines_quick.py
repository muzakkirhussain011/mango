#!/usr/bin/env python3
"""
Quick test to verify baseline algorithms produce different results.
Runs 3 rounds with each algorithm on Adult dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from faircare.experiments.run_experiments import FederatedExperiment

# Test configuration
config = {
    'algorithm': 'fedavg',  # Will be overridden
    'dataset': 'adult',
    'sensitive_attr': 'sex',
    'num_clients': 10,
    'rounds': 3,  # Just 3 rounds for quick test
    'local_epochs': 1,
    'learning_rate': 0.01,
    'batch_size': 128,
    'client_fraction': 0.3,
    'seed': 42,
    'save_dir': 'results/baseline_test',
    'experiment_name': 'baseline_test',
    'model': {
        'type': 'mlp',
        'hidden_dims': [256, 128]
    },
    'dirichlet_alpha': 0.3
}

algorithms = ['fedavg', 'fedprox', 'qffl', 'afl', 'fairfed']
results = {}

print("="*80)
print("BASELINE VALIDATION TEST")
print("="*80)
print(f"Running {config['rounds']} rounds with each baseline algorithm...")
print()

for algo in algorithms:
    print(f"Testing {algo}...")
    config['algorithm'] = algo
    config['experiment_name'] = f'baseline_test_{algo}'

    try:
        # Run experiment
        exp = FederatedExperiment(config)
        exp.prepare_data()
        exp.initialize_model()
        exp.initialize_aggregator()

        final_results = exp.run()

        # Store results (metrics are nested in 'final_metrics')
        test_metrics = final_results.get('final_metrics', {})
        results[algo] = {
            'accuracy': test_metrics.get('test/accuracy', 0.0),
            'worst_group_f1': test_metrics.get('test/worst_group_f1', 0.0),
            'eo_gap': test_metrics.get('test/eo_gap', 0.0)
        }

        print(f"  [OK] {algo}: acc={results[algo]['accuracy']:.4f}, wg_f1={results[algo]['worst_group_f1']:.4f}")

    except Exception as e:
        print(f"  [ERROR] {algo}: ERROR - {str(e)[:100]}")
        results[algo] = None

print()
print("="*80)
print("RESULTS SUMMARY")
print("="*80)

# Check if baselines are different
if all(v is not None for v in results.values()):
    accuracies = [results[algo]['accuracy'] for algo in algorithms]
    wg_f1s = [results[algo]['worst_group_f1'] for algo in algorithms]

    # Check variance
    acc_std = np.std(accuracies)
    wg_std = np.std(wg_f1s)

    print(f"\nAccuracy range: {min(accuracies):.4f} - {max(accuracies):.4f} (std: {acc_std:.4f})")
    print(f"Worst-Group F1 range: {min(wg_f1s):.4f} - {max(wg_f1s):.4f} (std: {wg_std:.4f})")

    if acc_std < 0.001 and wg_std < 0.001:
        print("\n[WARNING] Baselines still producing nearly identical results!")
        print("          This suggests the bug is not fully fixed.")
    else:
        print("\n[SUCCESS] Baselines are producing different results!")
        print("          The fix is working correctly.")
else:
    print("\n[WARNING] Some algorithms failed to run. Check errors above.")

print()
print("Detailed results:")
for algo, metrics in results.items():
    if metrics:
        print(f"  {algo:10s}: acc={metrics['accuracy']:.4f}, wg_f1={metrics['worst_group_f1']:.4f}, eo_gap={metrics['eo_gap']:.4f}")
    else:
        print(f"  {algo:10s}: FAILED")

print("="*80)
