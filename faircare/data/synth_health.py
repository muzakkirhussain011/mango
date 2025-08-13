"""Synthetic health data generator with controlled bias."""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split


class SyntheticHealthDataset(Dataset):
    """Synthetic health dataset with controlled bias."""
    
    def __init__(self, X, y, a):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.a = torch.LongTensor(a)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.a[idx]


def generate_synthetic_health(
    n_samples: int = 10000,
    n_features: int = 20,
    bias_level: float = 0.3,
    group_imbalance: float = 0.5,
    noise_level: float = 0.1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Generate synthetic health data with controlled bias.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        bias_level: Level of bias (0 = no bias, 1 = maximum bias)
        group_imbalance: Proportion of group 0 (0.5 = balanced)
        noise_level: Noise in features
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
    
    Returns:
        Dictionary with train/val/test data
    """
    np.random.seed(seed)
    
    # Generate sensitive attribute
    a = np.random.binomial(1, 1 - group_imbalance, n_samples)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Add group-specific patterns
    for i in range(n_samples):
        if a[i] == 0:
            X[i, :n_features//2] += 0.5  # Group 0 has higher values in first half
        else:
            X[i, n_features//2:] += 0.5  # Group 1 has higher values in second half
    
    # Generate labels with bias
    # Create a weight vector
    w = np.random.randn(n_features)
    
    # Compute base probabilities
    logits = X @ w
    base_probs = 1 / (1 + np.exp(-logits))
    
    # Add bias: group 0 has systematically lower positive rate
    biased_probs = base_probs.copy()
    for i in range(n_samples):
        if a[i] == 0:
            biased_probs[i] *= (1 - bias_level)
        else:
            biased_probs[i] = biased_probs[i] * (1 - bias_level) + bias_level
    
    # Generate labels
    y = np.random.binomial(1, biased_probs)
    
    # Add noise
    X += np.random.randn(n_samples, n_features) * noise_level
    
    # Split data
    X_temp, X_test, y_temp, y_test, a_temp, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=seed, stratify=y
    )
    
    X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
        X_temp, y_temp, a_temp, test_size=val_size/(1-test_size), 
        random_state=seed, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = SyntheticHealthDataset(X_train, y_train, a_train)
    val_dataset = SyntheticHealthDataset(X_val, y_val, a_val)
    test_dataset = SyntheticHealthDataset(X_test, y_test, a_test)
    
    # Compute statistics
    train_stats = {
        "n_group_0": (a_train == 0).sum(),
        "n_group_1": (a_train == 1).sum(),
        "pos_rate_group_0": y_train[a_train == 0].mean() if (a_train == 0).any() else 0,
        "pos_rate_group_1": y_train[a_train == 1].mean() if (a_train == 1).any() else 0,
    }
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "n_features": n_features,
        "n_classes": 2,
        "sensitive_attribute": "synthetic_group",
        "stats": train_stats,
        "generation_params": {
            "bias_level": bias_level,
            "group_imbalance": group_imbalance,
            "noise_level": noise_level
        }
    }
