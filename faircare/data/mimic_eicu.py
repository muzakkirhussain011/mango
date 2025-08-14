"""MIMIC and eICU dataset stubs with synthetic fallback."""
from typing import Dict, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
import warnings
from faircare.data.synth_health import generate_synthetic_health


class MIMICDataset(Dataset):
    """MIMIC dataset stub."""
    
    def __init__(self, X, y, a):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.a = torch.LongTensor(a) if a is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.a is not None:
            return self.X[idx], self.y[idx], self.a[idx]
        return self.X[idx], self.y[idx]


def load_mimic(
    data_path: Optional[str] = None,
    sensitive_attribute: Optional[str] = "gender",
    **kwargs
) -> Dict:
    """
    Load MIMIC-III/IV dataset.
    Falls back to synthetic data if not available.
    """
    if data_path is None:
        warnings.warn(
            "MIMIC data path not provided. Using synthetic data as fallback. "
            "To use real MIMIC data, provide data_path parameter."
        )
        
        # Generate synthetic data mimicking MIMIC characteristics
        return generate_synthetic_health(
            n_samples=20000,
            n_features=50,  # More features for ICU data
            bias_level=0.25,  # Moderate bias
            group_imbalance=0.45,  # Slight gender imbalance
            noise_level=0.15,
            **kwargs
        )
    
    # Placeholder for actual MIMIC loading
    # This would require proper credentialed access and preprocessing
    raise NotImplementedError(
        "Real MIMIC data loading requires credentialed access. "
        "Please use synthetic data for testing."
    )


def load_eicu(
    data_path: Optional[str] = None,
    sensitive_attribute: Optional[str] = "gender",
    **kwargs
) -> Dict:
    """
    Load eICU dataset.
    Falls back to synthetic data if not available.
    """
    if data_path is None:
        warnings.warn(
            "eICU data path not provided. Using synthetic data as fallback. "
            "To use real eICU data, provide data_path parameter."
        )
        
        # Generate synthetic data mimicking eICU characteristics
        return generate_synthetic_health(
            n_samples=15000,
            n_features=45,
            bias_level=0.2,
            group_imbalance=0.48,
            noise_level=0.12,
            **kwargs
        )
    
    # Placeholder for actual eICU loading
    raise NotImplementedError(
        "Real eICU data loading requires credentialed access. "
        "Please use synthetic data for testing."
    )
