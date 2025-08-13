#!/bin/bash
# Reproduce paper tables

set -e

echo "Reproducing paper tables..."
echo "=========================="

# Run main experiments
echo "Running experiments..."
python -m faircare.experiments.run_sweep \
    --config faircare/experiments/configs/search.yaml \
    --n_workers 4 \
    --output_dir results/tables

# Generate LaTeX tables
echo "Generating LaTeX tables..."
python -m paper.tables \
    --results_dir results/tables \
    --output_dir paper/tables

# Generate comparison tables
echo "Generating comparison tables..."
python -c "
import json
import pandas as pd
from pathlib import Path
from faircare.fairness.summarize import create_summary_table, export_latex_table

results_path = Path('results/tables/aggregated_results.json')
with open(results_path) as f:
    results = json.load(f)

# Main results table
df = create_summary_table(results)
export_latex_table(
    df,
    'paper/tables/main_results.tex',
    caption='Comparison of federated learning algorithms on fairness metrics',
    label='tab:main_results'
)

print('Tables generated successfully!')
"

echo ""
echo "Tables saved to: paper/tables/"
echo "=========================="
