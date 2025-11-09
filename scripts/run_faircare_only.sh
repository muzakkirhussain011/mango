#!/bin/bash
# Run FairCare-FL experiments only across all datasets

set -e  # Exit on error

echo "Starting FairCare-FL experimental evaluation..."
echo "======================================="

# Create results directory
mkdir -p results/full_evaluation

# FairCare-FL only
ALGORITHM="faircare_fl"
DATASETS=("adult" "heart" "synth_health")
SEEDS=(0 1 2 3 4)

for dataset in "${DATASETS[@]}"; do
    echo "Dataset: $dataset"
    echo "----------------"

    echo "  Algorithm: $ALGORITHM"

    for seed in "${SEEDS[@]}"; do
        echo "    Seed: $seed"

        python -m faircare.experiments.run_experiments \
            --algorithm $ALGORITHM \
            --dataset $dataset \
            --sensitive_attr sex \
            --num_clients 10 \
            --rounds 20 \
            --local_epochs 1 \
            --learning_rate 0.01 \
            --seed $seed \
            --save_dir results/full_evaluation/$ALGORITHM/$dataset/seed$seed
    done
done

echo ""
echo "======================================="
echo "FairCare-FL evaluation complete!"
echo "Results saved to: results/full_evaluation/faircare_fl/"
