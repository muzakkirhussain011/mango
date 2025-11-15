#!/bin/bash
# Comprehensive evaluation across ALL datasets and algorithms
# This script runs the complete experimental suite with FIXED baselines

set -e  # Exit on error

echo "=========================================================================="
echo "COMPREHENSIVE EVALUATION: All Algorithms Across All Datasets"
echo "=========================================================================="
echo ""
echo "This will run experiments for:"
echo "  - Algorithms: FedAvg, FedProx, q-FFL, AFL, FairFed, FairCare-FL"
echo "  - Datasets: Adult, COMPAS, MIMIC, eICU"
echo "  - Seeds: 0, 1, 2"
echo ""
echo "Total experiments: 6 algorithms x 4 datasets x 3 seeds = 72 experiments"
echo "Estimated time: 2-3 hours"
echo ""
echo "Starting experiments..."
echo ""

# Configuration
ALGORITHMS=("fedavg" "fedprox" "qffl" "afl" "fairfed" "faircare_fl")
DATASETS=("adult" "compas" "mimic" "eicu")
SEEDS=(0 1 2)

# Track progress
total_experiments=$((${#ALGORITHMS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
current_experiment=0
start_time=$(date +%s)

# Run experiments
for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: $dataset"
    echo "========================================"

    for algo in "${ALGORITHMS[@]}"; do
        echo "  Algorithm: $algo"

        for seed in "${SEEDS[@]}"; do
            current_experiment=$((current_experiment + 1))
            elapsed=$(($(date +%s) - start_time))

            echo "    [$current_experiment/$total_experiments] Seed: $seed (Elapsed: ${elapsed}s)"

            # Run experiment
            python -m faircare.experiments.run_experiments \
                --algorithm $algo \
                --dataset $dataset \
                --sensitive_attr sex \
                --num_clients 10 \
                --rounds 20 \
                --local_epochs 1 \
                --learning_rate 0.01 \
                --seed $seed \
                --save_dir results/full_evaluation/$dataset/$algo/seed$seed \
                2>&1 | grep -E "(Round [0-9]+/|Final test|EXPERIMENT COMPLETED)" || true

            echo "      [OK] Completed"
        done
    done
done

total_time=$(($(date +%s) - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo ""
echo "=========================================================================="
echo "All experiments completed!"
echo "=========================================================================="
echo ""
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved to: results/full_evaluation/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python scripts/analyze_comprehensive_results.py"
echo "  2. Review comparison tables and visualizations"
echo "  3. Assess FairCare-FL performance vs baselines"
echo ""
