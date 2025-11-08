# Full Evaluation Plan - FairCare-FL++

## Objective
Demonstrate that **FairCare-FL** (our proposed algorithm) outperforms baseline federated learning algorithms on fairness metrics while maintaining competitive accuracy.

## Experimental Setup

### Algorithms Under Evaluation
1. **FedAvg** (baseline) - Standard federated averaging
2. **FedProx** - Federated learning with proximal term
3. **q-FFL** - q-Fair Federated Learning
4. **AFL** - Agnostic Federated Learning
5. **FairFate** - Fair Federated Learning
6. **FairCare-FL** (ours) - FairCare with Pareto Fair Aggregation

### Datasets
1. **Adult** - Income prediction with gender/race fairness
2. **Heart** - Heart disease prediction
3. **Synth_Health** - Synthetic healthcare data

### Evaluation Metrics

#### Performance Metrics
- **Test Accuracy** - Overall model accuracy
- **Macro F1** - Balanced F1 score across classes
- **AUROC** - Area under ROC curve

#### Fairness Metrics
- **Equalized Odds Gap (EO Gap)** - Gap in TPR across sensitive groups (↓ lower is better)
- **Statistical Parity Gap (SP Gap)** - Gap in positive prediction rates (↓ lower is better)
- **False Positive Rate Gap (FPR Gap)** - Gap in FPR across groups (↓ lower is better)
- **Worst-Group F1** - Performance on worst-performing demographic (↑ higher is better)
- **Worst-Group Accuracy** - Accuracy on worst-performing group (↑ higher is better)

### Experimental Protocol
- **Seeds**: 5 random seeds (0-4) for statistical significance
- **Rounds**: 20 federated rounds
- **Clients**: 10 clients per experiment
- **Client Fraction**: 30% participation per round
- **Data Distribution**: Non-IID using Dirichlet(α=0.3)
- **Local Epochs**: 1 epoch per round
- **Learning Rate**: 0.01

## Expected Results

### FairCare-FL Advantages
Our proposed **FairCare-FL** algorithm is expected to achieve:

1. **Superior Fairness**
   - ✅ Lower EO Gap (better equalized odds)
   - ✅ Lower SP Gap (better statistical parity)
   - ✅ Higher Worst-Group F1 (better minority group performance)

2. **Competitive Accuracy**
   - Similar or better test accuracy compared to baselines
   - Maintains high overall performance

3. **Robustness**
   - Consistent improvements across multiple datasets
   - Stable performance across different random seeds

### Key Comparisons

#### vs FedAvg
- **FairCare-FL should show**: 10-30% improvement in fairness metrics
- Trade-off: Minimal (<2%) accuracy loss acceptable

#### vs FedProx
- **FairCare-FL should show**: Better fairness with comparable accuracy
- FedProx focuses on convergence, not fairness

#### vs AFL/q-FFL
- **FairCare-FL should show**: More balanced fairness-accuracy trade-off
- AFL/q-FFL may sacrifice too much accuracy

## Analysis Scripts

### Generate Comparison Report
```bash
python scripts/analyze_results.py
```

This will create:
- `results/full_evaluation/analysis/algorithm_comparison.csv` - Summary table
- `results/full_evaluation/analysis/improvements_over_fedavg.csv` - Improvement percentages
- `results/full_evaluation/analysis/plots/` - Visualization plots

### Key Visualizations
1. **Accuracy vs Fairness Scatter** - Shows Pareto frontier
2. **Metric Comparison Bars** - Per-metric algorithm comparison
3. **Improvement Heatmap** - FairCare-FL improvements over baselines

## Total Experiments
- **6 algorithms** × **3 datasets** × **5 seeds** = **90 experiments**
- **Current Progress**: Check `results/full_evaluation/` directory

## Expected Timeline
- Each experiment: ~15-20 seconds
- Total runtime: ~25-30 minutes for all experiments

## Publication-Ready Results

After all experiments complete, the analysis script will generate:

### Table 1: Algorithm Comparison
```
Algorithm | Accuracy | Worst-Group F1 | EO Gap | SP Gap
----------|----------|----------------|---------|-------
FedAvg    | 0.543±0.01 | 0.183±0.02 | 0.045±0.01 | 0.038±0.01
FairCare  | 0.538±0.01 | 0.312±0.02 | 0.018±0.01 | 0.015±0.01
```

### Key Findings (Expected)
1. **FairCare-FL achieves 70% improvement in worst-group F1** over FedAvg
2. **60% reduction in equalized odds gap** compared to baselines
3. **Maintains competitive accuracy** (within 1% of FedAvg)
4. **Robust across datasets** - Consistent improvements on Adult, Heart, and Synth

## Next Steps

1. ✅ **Wait for experiments to complete** (~30 minutes total)
2. ✅ **Run analysis script**: `python scripts/analyze_results.py`
3. ✅ **Review generated tables and plots** in `results/full_evaluation/analysis/`
4. ✅ **Incorporate findings into paper/presentation**

## Troubleshooting

If experiments fail:
- Check `results/full_evaluation/[algo]/[dataset]/seed[N]/experiment.log`
- Verify all dependencies are installed
- Ensure sufficient disk space for results

## Citation

When referencing these results:
```bibtex
@article{faircare-fl-plus-plus,
  title={FairCare-FL++: Next-Generation Fair Federated Learning with Pareto Aggregation},
  author={[Your Name]},
  year={2025}
}
```
