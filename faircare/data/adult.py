"""Adult Census Income dataset loader."""
from typing import Optional, Tuple, Dict

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from typing import Optional
import warnings
import os
from pathlib import Path
import urllib.request
import ssl


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


def download_adult_data(cache_file: Path) -> pd.DataFrame:
    """Download Adult dataset from UCI repository."""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Try to download training data
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    # Create SSL context to handle certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    dfs = []
    
    # Download training data
    try:
        print("Downloading Adult training data from UCI...")
        with urllib.request.urlopen(train_url, context=ssl_context) as response:
            train_data = response.read().decode('utf-8')
        
        # Parse the data
        lines = [line.strip() for line in train_data.split('\n') if line.strip()]
        data_rows = []
        for line in lines:
            if line:
                # Split by comma and strip whitespace
                row = [item.strip() for item in line.split(',')]
                if len(row) == len(column_names):
                    data_rows.append(row)
        
        df_train = pd.DataFrame(data_rows, columns=column_names)
        dfs.append(df_train)
        print(f"Downloaded {len(df_train)} training samples")
        
    except Exception as e:
        print(f"Failed to download training data: {e}")
    
    # Download test data
    try:
        print("Downloading Adult test data from UCI...")
        with urllib.request.urlopen(test_url, context=ssl_context) as response:
            test_data = response.read().decode('utf-8')
        
        # Parse test data (skip first line which is a comment)
        lines = [line.strip() for line in test_data.split('\n') if line.strip()]
        data_rows = []
        for line in lines[1:]:  # Skip first line
            if line:
                # Remove the trailing period from test labels
                line = line.rstrip('.')
                row = [item.strip() for item in line.split(',')]
                if len(row) == len(column_names):
                    data_rows.append(row)
        
        df_test = pd.DataFrame(data_rows, columns=column_names)
        dfs.append(df_test)
        print(f"Downloaded {len(df_test)} test samples")
        
    except Exception as e:
        print(f"Failed to download test data: {e}")
    
    if not dfs:
        raise ValueError("Failed to download any Adult data from UCI")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Save to cache
    df.to_csv(cache_file, index=False)
    print(f"Cached {len(df)} total samples to {cache_file}")
    
    return df


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
    cache_file = cache_dir / "adult_uci.csv"
    
    # Load from cache or download
    if cache_file.exists():
        print(f"Loading Adult data from cache: {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        print("Cache not found. Downloading Adult data from UCI repository...")
        df = download_adult_data(cache_file)
    
    print(f"Loaded {len(df)} samples")
    
    # Clean data
    df = df.replace('?', np.nan).dropna()
    print(f"After cleaning: {len(df)} samples")
    
    # Process income column (target variable)
    income_values = df['income'].astype(str).str.strip()
    
    # Create binary labels: >50K = 1, <=50K = 0
    y = ((income_values == '>50K') | (income_values == '>50K.')).astype(int).values
    
    print(f"Label distribution: {np.sum(y==0)} samples with <=50K, {np.sum(y==1)} samples with >50K")
    
    # Extract sensitive attribute BEFORE encoding
    if sensitive_attribute == "sex" and 'sex' in df.columns:
        sex_col = df['sex'].astype(str).str.strip()
        a = (sex_col == 'Male').astype(int).values
        print(f"Sensitive attribute (sex): {np.sum(a==0)} Female, {np.sum(a==1)} Male")
    elif sensitive_attribute == "race" and 'race' in df.columns:
        race_col = df['race'].astype(str).str.strip()
        a = (race_col == 'White').astype(int).values
        print(f"Sensitive attribute (race): {np.sum(a==0)} Non-White, {np.sum(a==1)} White")
    else:
        a = None
        print("No sensitive attribute")
    
    # Process features
    df_features = df.drop('income', axis=1)
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    
    for col in df_features.columns:
        # Try to convert to numeric
        try:
            pd.to_numeric(df_features[col])
            numeric_cols.append(col)
        except:
            categorical_cols.append(col)
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Process numeric columns
    X_numeric = df_features[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values if numeric_cols else np.array([])
    
    # Process categorical columns
    X_categorical_list = []
    for col in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df_features[col].astype(str).str.strip())
        X_categorical_list.append(encoded.reshape(-1, 1))
    
    # Combine features
    if X_categorical_list and X_numeric.size > 0:
        X_categorical = np.hstack(X_categorical_list)
        X = np.hstack([X_numeric, X_categorical])
    elif X_categorical_list:
        X = np.hstack(X_categorical_list)
    else:
        X = X_numeric
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.astype(float))
    
    # Verify data integrity
    assert len(X) == len(y), f"X and y lengths don't match: {len(X)} vs {len(y)}"
    if a is not None:
        assert len(a) == len(y), f"a and y lengths don't match: {len(a)} vs {len(y)}"
    
    # Split data
    if a is not None:
        # Split with stratification on y
        X_temp, X_test, y_temp, y_test, a_temp, a_test = train_test_split(
            X, y, a, test_size=test_size, random_state=seed, stratify=y
        )
        
        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_temp, y_temp, a_temp, test_size=val_size/(1-test_size), 
            random_state=seed, stratify=y_temp
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=seed, stratify=y_temp
        )
        a_train = a_val = a_test = None
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
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
