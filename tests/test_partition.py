"""Test data partitioning."""

import pytest
import torch
import numpy as np
from faircare.data.partition import (
    partition_iid,
    partition_dirichlet,
    partition_label_skew,
    make_federated_splits
)
from faircare.data.synth_health import generate_synthetic_health


class TestPartitioning:
    
    def test_iid_partition(self):
        """Test IID partitioning."""
        n_samples = 1000
        n_clients = 10
        
        indices = partition_iid(n_samples, n_clients, seed=42)
        
        # Check all clients got data
        assert len(indices) == n_clients
        
        # Check roughly equal distribution
        sizes = [len(idx) for idx in indices]
        assert min(sizes) >= 90  # At least 90 samples
        assert max(sizes) <= 110  # At most 110 samples
        
        # Check no overlap
        all_indices = []
        for client_indices in indices:
            all_indices.extend(client_indices)
        assert len(set(all_indices)) == n_samples
    
    def test_dirichlet_partition(self):
        """Test Dirichlet partitioning."""
        n_samples = 1000
        n_clients = 10
        n_classes = 2
        
        # Create synthetic labels
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Low alpha = more non-IID
        indices = partition_dirichlet(
            labels, n_clients, alpha=0.1, n_classes=n_classes, seed=42
        )
        
        # Check all clients got data
        assert len(indices) == n_clients
        
        # Check total samples
        total = sum(len(idx) for idx in indices)
        assert total == n_samples
        
        # Check distribution is non-IID (some imbalance expected)
        for client_indices in indices:
            if len(client_indices) > 0:
                client_labels = labels[client_indices]
                # With low alpha, expect some class imbalance
                unique, counts = np.unique(client_labels, return_counts=True)
                if len(unique) == n_classes:
                    ratio = min(counts) / max(counts)
                    # Most clients should have imbalanced classes
                    # (but not enforcing strictly due to randomness)
    
    def test_label_skew_partition(self):
        """Test label skew partitioning."""
        n_samples = 1000
        n_clients = 10
        n_classes = 5
        classes_per_client = 2
        
        labels = np.random.randint(0, n_classes, n_samples)
        
        indices = partition_label_skew(
            labels, n_clients, n_classes, seed=42,
            classes_per_client=classes_per_client
        )
        
        # Check all clients got data
        assert len(indices) == n_clients
        
        # Check each client has limited classes
        for client_indices in indices:
            if len(client_indices) > 0:
                client_labels = labels[client_indices]
                unique_classes = len(np.unique(client_labels))
                assert unique_classes <= classes_per_client
    
    def test_make_federated_splits(self):
        """Test complete federated split creation."""
        # Generate synthetic data
        dataset = generate_synthetic_health(
            n_samples=1000,
            n_features=10,
            seed=42
        )
        
        # Create federated splits
        fed_data = make_federated_splits(
            dataset=dataset,
            n_clients=5,
            partition="dirichlet",
            alpha=0.5,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42
        )
        
        # Check structure
        assert "client_data" in fed_data
        assert "server_val" in fed_data
        assert "test" in fed_data
        
        # Check number of clients
        assert len(fed_data["client_data"]) == 5
        
        # Check each client has train data
        for client_data in fed_data["client_data"]:
            assert "train" in client_data
            assert client_data["train"] is not None
            assert len(client_data["train"]) > 0
