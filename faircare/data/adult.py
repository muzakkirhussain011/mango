"""Adult Census Income dataset loader."""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
import openml
from typing import Optional, Tuple, Dict
import warnings
import os
from pathlib import Path


class AdultDataset(Dataset):
    """Adult dataset with sensitive attributes."""
    
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


def load_adult(
    sensitive_attribute: Optional[str] = "sex",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> Dict:
    """
    Load Adult Census Income dataset.
    
    Args:
        sensitive_attribute: "sex", "race", or None
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
    cache_file = cache_dir / "adult.csv"
    
    # Try OpenML first
    try:
        # OpenML dataset ID for adult
        dataset = openml.datasets.get_dataset(1590, download_data=True)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )
        df = pd.concat([X, y], axis=1)
        
    except Exception as e:
        warnings.warn(f"OpenML failed: {e}. Trying fallback...")
        
        # Fallback to UCI repository
        if cache_file.exists():
            df = pd.read_csv(cache_file)
        else:
            # Download from UCI
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
            ]
            
            df = pd.read_csv(url, names=column_names, sep=',\s*', engine='python')
            
            # Save to cache
            df.to_csv(cache_file, index=False)
    
    # Clean data
    df = df.replace('?', np.nan).dropna()
    
    # Encode target variable
    y = (df['income'].str.strip() == '>50K').astype(int).values
    
    # Extract sensitive attribute
    if sensitive_attribute == "sex":
        a = (df['sex'].str.strip() == 'Male').astype(int).values
    elif sensitive_attribute == "race":
        a = (df['race'].str.strip() == 'White').astype(int).values
    else:
        a = None
    
    # Prepare features
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.drop(['income'])
    
    # Encode categorical variables
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Select features
    feature_columns = df_encoded.columns.drop(['income'])
    X = df_encoded[feature_columns].values
    
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
    train_dataset = AdultDataset(X_train, y_train, a_train)
    val_dataset = AdultDataset(X_val, y_val, a_val)
    test_dataset = AdultDataset(X_test, y_test, a_test)
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "n_features": X.shape[1],
        "n_classes": 2,
        "sensitive_attribute": sensitive_attribute
    }
