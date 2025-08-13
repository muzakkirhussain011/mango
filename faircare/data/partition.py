"""Data partitioning utilities for federated learning."""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


def make_federated_splits(
    dataset: Dict,
    n_clients: int,
    partition: str = "dirichlet",
    alpha: float = 0.5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create federated data splits.
    
    Args:
        dataset: Dataset dictionary from load_dataset
        n_clients: Number of clients
        partition: Partition strategy ("iid", "dirichlet", "label_skew")
        alpha: Dirichlet concentration parameter
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
    
    Returns:
        Dictionary with client data and server validation/test sets
    """
    np.random.seed(seed)
    
    # Extract full dataset
    train_data = dataset["train"]
    val_data = dataset.get("val")
    test_data = dataset.get("test")
    
    # Get data tensors
    if hasattr(train_data, 'X'):
        X_train = train_data.X
        y_train = train_data.y
        a_train = train_data.a if hasattr(train_data, 'a') else None
    else:
        # Handle different dataset formats
        all_data = []
        for i in range(len(train_data)):
            all_data.append(train_data[i])
        
        if len(all_data[0]) == 3:
            X_train = torch.stack([d[0] for d in all_data])
            y_train = torch.stack([d[1] for d in all_data])
            a_train = torch.stack([d[2] for d in all_data])
        else:
            X_train = torch.stack([d[0] for d in all_data])
            y_train = torch.stack([d[1] for d in all_data])
            a_train = None
    
    n_samples = len(X_train)
    n_classes = len(torch.unique(y_train))
    
    # Create client indices based on partition strategy
    if partition == "iid":
        client_indices = partition_iid(n_samples, n_clients, seed)
    elif partition == "dirichlet":
        client_indices = partition_dirichlet(
            y_train.numpy(), n_clients, alpha, n_classes, seed
        )
    elif partition == "label_skew":
        client_indices = partition_label_skew(
            y_train.numpy(), n_clients, n_classes, seed
        )
    else:
        raise ValueError(f"Unknown partition strategy: {partition}")
    
    # Create client datasets
    client_data = []
    server_val_X = []
    server_val_y = []
    server_val_a = []
    
    for client_id in range(n_clients):
        indices = client_indices[client_id]
        
        # Split client data into train/val
        n_client_samples = len(indices)
        n_val = int(n_client_samples * val_ratio)
        
        np.random.shuffle(indices)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Create client train dataset
        if a_train is not None:
            client_train = TensorDataset(
                X_train[train_indices],
                y_train[train_indices],
                a_train[train_indices]
            )
            client_val = TensorDataset(
                X_train[val_indices],
                y_train[val_indices],
                a_train[val_indices]
            ) if n_val > 0 else None
            
            # Collect for server validation
            if n_val > 0:
                server_val_X.append(X_train[val_indices])
                server_val_y.append(y_train[val_indices])
                server_val_a.append(a_train[val_indices])
        else:
            client_train = TensorDataset(
                X_train[train_indices],
                y_train[train_indices]
            )
            client_val = TensorDataset(
                X_train[val_indices],
                y_train[val_indices]
            ) if n_val > 0 else None
            
            if n_val > 0:
                server_val_X.append(X_train[val_indices])
                server_val_y.append(y_train[val_indices])
        
        client_data.append({
            "train": client_train,
            "val": client_val,
            "n_samples": len(train_indices)
        })
    
    # Create server validation set
    if server_val_X:
        server_val = (
            torch.cat(server_val_X),
            torch.cat(server_val_y),
            torch.cat(server_val_a) if server_val_a else None
        )
    else:
        server_val = None
    
    # Prepare test set
    if test_data is not None:
        if hasattr(test_data, 'X'):
            test_tuple = (test_data.X, test_data.y, test_data.a if hasattr(test_data, 'a') else None)
        else:
            # Extract from dataset
            test_X = []
            test_y = []
            test_a = []
            for i in range(len(test_data)):
                item = test_data[i]
                test_X.append(item[0])
                test_y.append(item[1])
                if len(item) > 2:
                    test_a.append(item[2])
            
            test_tuple = (
                torch.stack(test_X),
                torch.stack(test_y),
                torch.stack(test_a) if test_a else None
            )
    else:
        test_tuple = None
    
    return {
        "client_data": client_data,
        "server_val": server_val,
        "test": test_tuple,
        "n_clients": n_clients,
        "partition": partition,
        "partition_params": {"alpha": alpha} if partition == "dirichlet" else {}
    }


def partition_iid(
    n_samples: int,
    n_clients: int,
    seed: int
) -> List[np.ndarray]:
    """IID partition."""
    np.random.seed(seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split evenly
    splits = np.array_split(indices, n_clients)
    return [split.tolist() for split in splits]


def partition_dirichlet(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    n_classes: int,
    seed: int
) -> List[List[int]]:
    """
    Dirichlet-based non-IID partition.
    
    Lower alpha = more non-IID (more skewed distributions).
    """
    np.random.seed(seed)
    n_samples = len(labels)
    
    # Get indices for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)
    
    # Sample from Dirichlet distribution
    client_indices = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        indices = np.array(class_indices[c])
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * n_clients)
        
        # Normalize and compute splits
        proportions = proportions / proportions.sum()
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        # Assign to clients
        start = 0
        for client_id, prop in enumerate(proportions):
            if prop > 0:
                client_indices[client_id].extend(
                    indices[start:start + prop].tolist()
                )
            start += prop
    
    return client_indices


def partition_label_skew(
    labels: np.ndarray,
    n_clients: int,
    n_classes: int,
    seed: int,
    classes_per_client: int = 2
) -> List[List[int]]:
    """
    Label skew partition where each client has limited classes.
    """
    np.random.seed(seed)
    
    # Get indices for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)
    
    client_indices = [[] for _ in range(n_clients)]
    
    # Assign classes to clients
    for client_id in range(n_clients):
        # Select random classes for this client
        selected_classes = np.random.choice(
            n_classes,
            min(classes_per_client, n_classes),
            replace=False
        )
        
        for class_id in selected_classes:
            indices = class_indices[class_id]
            # Give portion of this class to client
            n_samples_for_client = len(indices) // n_clients
            
            if n_samples_for_client > 0:
                selected = np.random.choice(
                    indices,
                    n_samples_for_client,
                    replace=False
                )
                client_indices[client_id].extend(selected.tolist())
    
    return client_indices
