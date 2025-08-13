#!/bin/bash
# Run all experiments for paper reproduction

set -e  # Exit on error

echo "Starting full experimental evaluation..."
echo "======================================="

# Create results directory
mkdir -p results/full_evaluation

# Run experiments for each algorithm and dataset
ALGORITHMS=("fedavg" "fedprox" "qffl" "afl" "fairfate" "faircare_fl")
DATASETS=("adult" "heart" "synth_health")
SEEDS=(0 1 2 3 4)

for dataset in "${DATASETS[@]}"; do
    echo "Dataset: $dataset"
    echo "----------------"
    
    for algo in "${ALGORITHMS[@]}"; do
        echo "  Algorithm: $algo"
        
        for seed in "${SEEDS[@]}"; do
            echo "    Seed: $seed"
            
            python -m faircare.experiments.run_experiments \
                --algo $algo \
                --dataset $dataset \
                --sensitive sex \
                --clients 10 \
                --rounds 20 \
                --local_epochs 1 \
                --lr 0.01 \
                --seed $seed \
                --logdir results/full_evaluation/$algo/$dataset/seed$seed \
                --device cpu
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
