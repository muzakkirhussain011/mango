# MANGO: Fair Federated Learning for Healthcare Bias Mitigation

A production-quality, research-grade framework for fair federated learning with a focus on healthcare applications.

## Features

- **Algorithms**: FedAvg, FedProx, AFL, q-FFL, FAIR-FATE, FairCare-FL
- **Fairness Metrics**: Demographic Parity, Equal Opportunity, Equalized Odds
- **Secure Aggregation**: Research stub with cryptographic backend interface
- **Datasets**: Adult, Heart, Synthetic Health, MIMIC/eICU stubs
- **Full Reproducibility**: Seeded experiments, configuration management

## Quick Start (Google Colab)

```bash
# Clone repository
%cd /content
!git clone https://github.com/your-org/mango.git
%cd mango

# Install
!pip install -e . -q

# Set cache directory for OpenML
!mkdir -p /content/openml_cache
%env OPENML_CACHE_DIR=/content/openml_cache

# Run tests
!pytest -q

# Single experiment
!python -m faircare.experiments.run_experiments \
  --algo faircare_fl --dataset adult --sensitive sex \
  --clients 10 --rounds 20 --eval_every 1 \
  --seed 0 --logdir runs/faircare_fl/adult/seed0

# Full sweep
!python -m faircare.experiments.run_sweep --config faircare/experiments/configs/search.yaml
!python -m paper.tables
!python -m paper.make_figures
```

## Citations

- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (arXiv:1602.05629)
- **AFL**: Mohri et al., "Agnostic Federated Learning" (PMLR 97:4615-4625, 2019)
- **q-FFL**: Li et al., "Fair Resource Allocation in Federated Learning" (arXiv:1905.10497)
- **FAIR-FATE**: Ezzeldin et al., "FairFed: Enabling Group Fairness in Federated Learning" (arXiv:2110.00857)
- **Fairness Metrics**: Hardt et al., "Equality of Opportunity in Supervised Learning" (NeurIPS 2016)
- **Secure Aggregation**: Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017)
- **Fairlearn**: Bird et al., "Fairlearn: A toolkit for assessing and improving fairness in AI" (fairlearn.org)

## License

MIT License

