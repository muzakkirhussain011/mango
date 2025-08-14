from typing import Dict, List, Optional, Tuple, Any, Union, Protocol

"""Test aggregator implementations."""

import pytest
import torch
from faircare.algos import make_aggregator


class TestAggregators:
    
    def test_fedavg_weights(self):
        """Test FedAvg weight computation."""
        aggregator = make_aggregator("fedavg", n_clients=3)
        
        client_summaries = [
            {"n_samples": 100, "train_loss": 1.0},
            {"n_samples": 200, "train_loss": 0.8},
            {"n_samples": 300, "train_loss": 0.6}
        ]
        
        weights = aggregator.compute_weights(client_summaries)
        
        # Weights should be proportional to n_samples
        expected = torch.tensor([100, 200, 300], dtype=torch.float32)
        expected = expected / expected.sum()
        
        assert torch.allclose(weights, expected)
    
    def test_qffl_weights(self):
        """Test q-FFL weight computation."""
        aggregator = make_aggregator("qffl", n_clients=3, q=2.0)
        
        client_summaries = [
            {"train_loss": 1.0},
            {"train_loss": 2.0},
            {"train_loss": 0.5}
        ]
        
        weights = aggregator.compute_weights(client_summaries)
        
        # Higher loss should get higher weight with q > 1
        assert weights[1] > weights[0] > weights[2]
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
    
    def test_faircare_weights(self):
        """Test FairCare-FL weight computation."""
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=3,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            tau=1.0
        )
        
        client_summaries = [
            {"eo_gap": 0.1, "fpr_gap": 0.0, "sp_gap": 0.0, "val_loss": 1.0},
            {"eo_gap": 0.3, "fpr_gap": 0.0, "sp_gap": 0.0, "val_loss": 1.0},
            {"eo_gap": 0.2, "fpr_gap": 0.0, "sp_gap": 0.0, "val_loss": 1.0}
        ]
        
        weights = aggregator.compute_weights(client_summaries)
        
        # Lower EO gap should get higher weight
        assert weights[0] > weights[2] > weights[1]
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
    
    def test_weight_floor(self):
        """Test weight floor in FairCare-FL."""
        aggregator = make_aggregator(
            "faircare_fl",
            n_clients=5,
            epsilon=0.1
        )
        
        # Create very imbalanced scores
        client_summaries = [
            {"eo_gap": 0.0, "fpr_gap": 0.0, "sp_gap": 0.0, "val_loss": 0.0},
            {"eo_gap": 10.0, "fpr_gap": 10.0, "sp_gap": 10.0, "val_loss": 10.0},
            {"eo_gap": 10.0, "fpr_gap": 10.0, "sp_gap": 10.0, "val_loss": 10.0},
            {"eo_gap": 10.0, "fpr_gap": 10.0, "sp_gap": 10.0, "val_loss": 10.0},
            {"eo_gap": 10.0, "fpr_gap": 10.0, "sp_gap": 10.0, "val_loss": 10.0}
        ]
        
        weights = aggregator.compute_weights(client_summaries)
        
        # All weights should be at least epsilon
        assert (weights >= 0.1).all()
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
