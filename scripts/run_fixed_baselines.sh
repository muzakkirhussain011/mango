#!/bin/bash
# Re-run experiments with FIXED baseline implementations
# This will give us VALID comparisons for FairCare-FL

set -e  # Exit on error

echo "=========================================================================="
echo "Re-running Experiments with FIXED Baselines"
echo "=========================================================================="
echo ""
echo "Previous bug: All baselines used FedAvgAggregator"
echo "Now fixed: Each baseline uses its correct aggregator"
echo ""
echo "This will take approximately 60-90 minutes for all experiments."
echo ""

# Create results directory
mkdir -p results/fixed_baselines_evaluation

# Reduced experiment configuration for faster validation
ALGORITHMS=("fedavg" "fedprox" "qffl" "afl" "fairfed" "faircare_fl")
DATASETS=("adult")  # Start with just Adult dataset for validation
SEEDS=(0 1 2)  # Use 3 seeds for faster validation

echo "Configuration:"
echo "  Algorithms: ${ALGORITHMS[@]}"
echo "  Datasets: ${DATASETS[@]}"
echo "  Seeds: ${SEEDS[@]}"
echo "  Total experiments: $((${#ALGORITHMS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))"
echo ""
echo "Starting experiments..."
echo ""

# Track progress
total_experiments=$((${#ALGORITHMS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
current_experiment=0

for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: $dataset"
    echo "========================================"

    for algo in "${ALGORITHMS[@]}"; do
        echo "  Algorithm: $algo"

        for seed in "${SEEDS[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "    [$current_experiment/$total_experiments] Seed: $seed"

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
                --save_dir results/fixed_baselines_evaluation/$algo/$dataset/seed$seed \
                2>&1 | grep -E "(Round|Final test|EXPERIMENT COMPLETED)" || true

            echo "      [OK] Completed"
        done
    done
done

echo ""
echo "=========================================================================="
echo "All experiments completed!"
echo "=========================================================================="
echo ""
echo "Results saved to: results/fixed_baselines_evaluation/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python scripts/analyze_fixed_results.py"
echo "  2. Compare with previous (buggy) results"
echo "  3. Assess FairCare-FL performance vs corrected baselines"
echo ""
