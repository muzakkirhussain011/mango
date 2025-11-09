#!/bin/bash
# Run all experiments for paper reproduction

set -e  # Exit on error

echo "Starting full experimental evaluation..."
echo "======================================="

# Create results directory
mkdir -p results/full_evaluation

# Run experiments for each algorithm and dataset
ALGORITHMS=("fedavg" "fedprox" "qffl" "afl" "fairfed" "faircare_fl")
DATASETS=("adult" "compas" "synthetic")
SEEDS=(0 1 2 3 4)

for dataset in "${DATASETS[@]}"; do
    echo "Dataset: $dataset"
    echo "----------------"
    
    for algo in "${ALGORITHMS[@]}"; do
        echo "  Algorithm: $algo"
        
        for seed in "${SEEDS[@]}"; do
            echo "    Seed: $seed"
            
            python -m faircare.experiments.run_experiments \
                --algorithm $algo \
                --dataset $dataset \
                --sensitive_attr sex \
                --num_clients 10 \
                --rounds 20 \
                --local_epochs 1 \
                --learning_rate 0.01 \
                --seed $seed \
                --save_dir results/full_evaluation/$algo/$dataset/seed$seed
        done
    done
done

echo ""
echo "Running statistical analysis..."
python -m faircare.experiments.run_sweep \
    --config faircare/experiments/configs/search.yaml \
    --output_dir results/full_evaluation/summary

echo ""
echo "Generating tables and figures..."
python -m paper.tables
python -m paper.make_figures

echo ""
echo "======================================="
echo "Evaluation complete!"
echo "Results saved to: results/full_evaluation/"
