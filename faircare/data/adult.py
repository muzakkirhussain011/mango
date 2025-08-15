"""Adult Census Income dataset loader."""
from typing import Optional, Tuple, Dict

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
import openml
from typing import Optional
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
        
        # Find the target column name (it might be 'class' or something else)
        target_col = None
        for col in df.columns:
            if 'class' in col.lower() or 'income' in col.lower() or 'target' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            # Last column is often the target
            target_col = df.columns[-1]
        
        # Rename target column to 'income' for consistency
        df = df.rename(columns={target_col: 'income'})
        
    except Exception as e:
        warnings.warn(f"OpenML failed: {e}. Using fallback...")
        
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
            
            try:
                df = pd.read_csv(url, names=column_names, sep=',\s*', engine='python')
            except:
                # If UCI fails, generate synthetic data as fallback
                warnings.warn("Both OpenML and UCI failed. Using synthetic data as fallback.")
                from faircare.data.synth_health import generate_synthetic_health
                return generate_synthetic_health(
                    n_samples=30000,
                    n_features=108,  # Adult usually has 108 features after encoding
                    bias_level=0.3,
                    group_imbalance=0.5,
                    seed=seed
                )
            
            # Save to cache
            df.to_csv(cache_file, index=False)
    
    # Clean data
    df = df.replace('?', np.nan).dropna()
    
    # Encode target variable - handle different formats
    if 'income' in df.columns:
        income_col = df['income'].astype(str).str.strip()
        # Handle both '>50K' and '1' formats
        if any('>50K' in str(v) for v in income_col.unique()):
            y = (income_col == '>50K').astype(int).values
        elif any('<=50K' in str(v) for v in income_col.unique()):
            y = (income_col != '<=50K').astype(int).values
        else:
            # If it's already numeric or different format
            y_numeric = pd.to_numeric(income_col, errors='coerce')
            if not y_numeric.isna().all():
                y = (y_numeric > 0).astype(int).values
            else:
                # If conversion failed, try to interpret as binary
                unique_vals = income_col.unique()
                if len(unique_vals) == 2:
                    y = (income_col == unique_vals[1]).astype(int).values
                else:
                    y = (income_col == income_col.mode()[0]).astype(int).values
    else:
        # If no income column, use synthetic data
        warnings.warn("No income column found. Using synthetic data.")
        from faircare.data.synth_health import generate_synthetic_health
        return generate_synthetic_health(
            n_samples=30000,
            n_features=108,
            bias_level=0.3,
            group_imbalance=0.5,
            seed=seed
        )
    
    # Extract sensitive attribute BEFORE encoding
    if sensitive_attribute == "sex" and 'sex' in df.columns:
        sex_col = df['sex'].astype(str).str.strip()
        a = (sex_col == 'Male').astype(int).values
    elif sensitive_attribute == "race" and 'race' in df.columns:
        race_col = df['race'].astype(str).str.strip()
        a = (race_col == 'White').astype(int).values
    else:
        a = None
    
    # Now encode ALL columns (except income which we already processed)
    df_encoded = df.copy()
    if 'income' in df_encoded.columns:
        df_encoded = df_encoded.drop('income', axis=1)
    
    # Separate numeric and categorical columns
    numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # Convert to string first to handle any mixed types
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    # Make sure all columns are numeric now
    X = df_encoded.values.astype(float)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Ensure we have the right lengths
    assert len(X) == len(y), f"X and y lengths don't match: {len(X)} vs {len(y)}"
    if a is not None:
        assert len(a) == len(y), f"a and y lengths don't match: {len(a)} vs {len(y)}"
    
    # Split data
    if a is not None:
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
