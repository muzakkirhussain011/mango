# ============================================================================
# faircare/data/datasets.py
# ============================================================================

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from typing import List, Tuple, Dict, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FairDataset(Dataset):
    """Dataset wrapper with sensitive attributes."""
    
    def __init__(self, features, labels, sensitive_attrs):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sensitive_attrs = torch.LongTensor(sensitive_attrs)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.sensitive_attrs[idx]


def load_dataset(name: str, task: str, sensitive_attr: str, 
                preprocessing: Dict = None, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and preprocess dataset.
    
    Args:
        name: Dataset name
        task: Prediction task
        sensitive_attr: Sensitive attribute name
        preprocessing: Preprocessing configuration
        
    Returns:
        Train, validation, and test datasets
    """
    # Synthetic data generation for demonstration
    np.random.seed(42)
    
    if name == 'mimic':
        n_samples = 10000
        n_features = 50
    elif name == 'eicu':
        n_samples = 8000
        n_features = 45
    elif name == 'adult':
        n_samples = 15000
        n_features = 30
    elif name == 'compas':
        n_samples = 5000
        n_features = 20
    else:
        n_samples = 5000
        n_features = 20
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with some correlation to features
    weights = np.random.randn(n_features)
    logits = X @ weights + np.random.randn(n_samples) * 0.5
    y = (logits > 0).astype(int)
    
    # Generate sensitive attributes (binary for simplicity)
    sensitive = np.random.binomial(1, 0.4, n_samples)
    
    # Add correlation between sensitive attribute and outcome (for bias)
    y[sensitive == 1] = np.random.binomial(1, 0.7, (sensitive == 1).sum())
    y[sensitive == 0] = np.random.binomial(1, 0.3, (sensitive == 0).sum())
    
    # Preprocessing
    if preprocessing and preprocessing.get('normalize', False):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Split data
    X_temp, X_test, y_temp, y_test, s_temp, s_test = train_test_split(
        X, y, sensitive, test_size=0.1, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.111, random_state=42, stratify=y_temp
    )
    
    # Create datasets
    train_dataset = FairDataset(X_train, y_train, s_train)
    val_dataset = FairDataset(X_val, y_val, s_val)
    test_dataset = FairDataset(X_test, y_test, s_test)
    
    return train_dataset, val_dataset, test_dataset


def create_federated_splits(dataset: Dataset, num_clients: int, 
                          alpha: float = 0.5, seed: int = 42) -> Dict[int, Dataset]:
    """Create federated data splits using Dirichlet distribution.
    
    Args:
        dataset: Original dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed
        
    Returns:
        Dictionary mapping client ID to dataset
    """
    np.random.seed(seed)
    
    # Get all data
    all_features = []
    all_labels = []
    all_sensitive = []
    
    for i in range(len(dataset)):
        features, label, sensitive = dataset[i]
        all_features.append(features)
        all_labels.append(label)
        all_sensitive.append(sensitive)
    
    all_features = torch.stack(all_features)
    all_labels = torch.stack(all_labels)
    all_sensitive = torch.stack(all_sensitive)
    
    # Get unique labels
    unique_labels = torch.unique(all_labels)
    n_classes = len(unique_labels)
    
    # Create label-wise indices
    label_indices = {}
    for label in unique_labels:
        label_indices[label.item()] = torch.where(all_labels == label)[0].numpy()
    
    # Distribute data using Dirichlet
    client_data = {i: [] for i in range(num_clients)}
    
    for label in unique_labels:
        indices = label_indices[label.item()]
        np.random.shuffle(indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(indices)).astype(int)
        proportions[proportions.argmax()] += len(indices) - proportions.sum()
        
        # Assign indices to clients
        start = 0
        for client_id, num_samples in enumerate(proportions):
            if num_samples > 0:
                client_indices = indices[start:start + num_samples]
                client_data[client_id].extend(client_indices)
                start += num_samples
    
    # Create client datasets
    client_datasets = {}
    for client_id in range(num_clients):
        if client_data[client_id]:
            indices = torch.tensor(client_data[client_id])
            client_features = all_features[indices]
            client_labels = all_labels[indices]
            client_sensitive = all_sensitive[indices]
            
            client_datasets[client_id] = FairDataset(
                client_features, client_labels, client_sensitive
            )
    
    return client_datasets

