"""Test secure aggregation."""

import pytest
import torch
import numpy as np
from faircare.core.secure_agg import SecureAggregator


class TestSecureAggregation:
    
    def test_additive_masking(self):
        """Test additive masking protocol."""
        n_clients = 5
        aggregator = SecureAggregator(
            n_clients=n_clients,
            protocol="additive_masking"
        )
        
        # Create client updates
        updates = []
        for i in range(n_clients):
            update = {
                "weight": torch.randn(10, 5),
                "bias": torch.randn(5)
            }
            updates.append(update)
        
        # Aggregate
        weights = torch.ones(n_clients) / n_clients
        result = aggregator.aggregate(updates, weights)
        
        # Check shapes
        assert result["weight"].shape == (10, 5)
        assert result["bias"].shape == (5,)
        
        # Without masks, result should be average
        # (masks should sum to zero)
        expected_weight = sum(u["weight"] for u in updates) / n_clients
        expected_bias = sum(u["bias"] for u in updates) / n_clients
        
        # Due to masking, exact equality won't hold, but should be close
        assert torch.allclose(result["weight"], expected_weight, atol=0.1)
        assert torch.allclose(result["bias"], expected_bias, atol=0.1)
    
    def test_secure_sum_tensors(self):
        """Test secure sum of tensors."""
        aggregator = SecureAggregator(
            n_clients=3,
            protocol="additive_masking"
        )
        
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor([7.0, 8.0, 9.0])
        ]
        
        weights = [0.2, 0.3, 0.5]
        
        result = aggregator.secure_sum_tensors(tensors, weights)
        
        # Check shape
        assert result.shape == (3,)
        
        # Result should be weighted sum (approximately)
        expected = sum(w * t for w, t in zip(weights, tensors))
        assert torch.allclose(result, expected, atol=0.1)
    
    def test_plain_aggregation(self):
        """Test plain aggregation without security."""
        aggregator = SecureAggregator(
            n_clients=3,
            protocol="none"
        )
        
        updates = [
            {"param": torch.tensor([1.0, 2.0])},
            {"param": torch.tensor([3.0, 4.0])},
            {"param": torch.tensor([5.0, 6.0])}
        ]
        
        weights = torch.tensor([0.5, 0.3, 0.2])
        result = aggregator.aggregate(updates, weights)
        
        expected = (
            0.5 * torch.tensor([1.0, 2.0]) +
            0.3 * torch.tensor([3.0, 4.0]) +
            0.2 * torch.tensor([5.0, 6.0])
        )
        
        assert torch.allclose(result["param"], expected)
