"""Heart Disease dataset loader."""
from typing import Optional, Dict

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Optional
import warnings
from pathlib import Path


class HeartDataset(Dataset):
    """Heart disease dataset with sensitive attributes."""
    
    def __init__(self, X, y, a=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.a = torch.LongTensor(a) if a is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.a is not None:
            return self.X[idx], self.y[idx], self.a[idx]
        return self.X[idx], self.y[idx]


def load_heart(
    sensitive_attribute: Optional[str] = "sex",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> Dict:
    """
    Load Heart Disease dataset from UCI.
    
    Args:
        sensitive_attribute: "sex", "age", or None
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
        cache_dir: Directory for caching data
    
    Returns:
        Dictionary with train/val/test data
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".faircare" / "data"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "heart.csv"
    
    # Load or download data
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        # Download from UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        df = pd.read_csv(url, names=column_names)
        
        # Clean data
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Save to cache
        df.to_csv(cache_file, index=False)
    
    # Prepare target (binary: 0 = no disease, 1 = disease)
    y = (df['target'] > 0).astype(int).values
    
    # Extract sensitive attribute
    if sensitive_attribute == "sex":
        a = df['sex'].values.astype(int)  # 1 = male, 0 = female
    elif sensitive_attribute == "age":
        # Binary: above/below median age
        median_age = df['age'].median()
        a = (df['age'] > median_age).astype(int).values
    else:
        a = None
    
    # Prepare features
    feature_columns = df.columns.drop(['target'])
    X = df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    if a is not None:
        a_temp = a[: len(X_temp)]
        a_test = a[len(X_temp): len(X_temp) + len(X_test)]
    else:
        a_temp = None
        a_test = None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=seed, stratify=y_temp
    )
    
    if a is not None:
        a_train = a_temp[: len(X_train)]
        a_val = a_temp[len(X_train):]
    else:
        a_train = None
        a_val = None
    
    # Create datasets
    train_dataset = HeartDataset(X_train, y_train, a_train)
    val_dataset = HeartDataset(X_val, y_val, a_val)
    test_dataset = HeartDataset(X_test, y_test, a_test)
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "n_features": X.shape[1],
        "n_classes": 2,
        "sensitive_attribute": sensitive_attribute
    }
