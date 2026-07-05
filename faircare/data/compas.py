"""COMPAS recidivism dataset loader (ProPublica).

Real, freely-downloadable dataset from ProPublica's "Machine Bias" investigation.
Task: predict two-year recidivism. Sensitive attribute: race or sex.

We apply ProPublica's standard filtering and do NOT use the COMPAS decile score / score_text as
features (those ARE the proprietary risk prediction we are studying, not inputs to our model).

Reference: Angwin, Larson, Mattu, Kirchner, "Machine Bias", ProPublica, 2016.
Data: https://github.com/propublica/compas-analysis
"""
from typing import Optional, Dict
import ssl
import urllib.request
import warnings
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)

# Features used to predict recidivism (excludes COMPAS's own score to avoid label leakage).
_FEATURE_COLS = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count",
    "c_charge_degree", "age_cat", "sex", "race",
]
_CATEGORICAL = ["c_charge_degree", "age_cat", "sex", "race"]


class CompasDataset(Dataset):
    """COMPAS dataset with a binary sensitive attribute."""

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


def _download_compas(cache_file: Path) -> pd.DataFrame:
    print("Downloading COMPAS data from ProPublica...")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(_COMPAS_URL, context=ssl_context) as response:
        raw = response.read().decode("utf-8")
    from io import StringIO
    df = pd.read_csv(StringIO(raw))
    df.to_csv(cache_file, index=False)
    print(f"Cached {len(df)} rows to {cache_file}")
    return df


def load_compas(
    sensitive_attribute: Optional[str] = "race",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Load the ProPublica COMPAS two-year recidivism dataset.

    Args:
        sensitive_attribute: "race" (Caucasian=1 vs non-Caucasian=0), "sex" (Male=1, Female=0), or None.
        test_size / val_size: split fractions.
        seed: random seed.
        cache_dir: cache directory (default ~/.faircare/data).

    Returns:
        Dict with train/val/test datasets and metadata (matches the repo's loader contract).
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".faircare" / "data"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "compas_two_years.csv"

    if cache_file.exists():
        print(f"Loading COMPAS data from cache: {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        df = _download_compas(cache_file)

    # ProPublica's standard filtering.
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    # Target: recidivism within two years.
    y = df["two_year_recid"].astype(int).values

    # Sensitive attribute (extract before encoding).
    if sensitive_attribute == "race":
        a = (df["race"].astype(str).str.strip() == "Caucasian").astype(int).values  # 1=Caucasian
    elif sensitive_attribute in ("sex", "gender"):
        a = (df["sex"].astype(str).str.strip() == "Male").astype(int).values
    else:
        a = None

    # Build feature matrix (keep sensitive columns in features, mirroring adult.py).
    df_features = df[_FEATURE_COLS].copy()
    numeric_cols = [c for c in _FEATURE_COLS if c not in _CATEGORICAL]
    parts = [df_features[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values]
    for col in _CATEGORICAL:
        le = LabelEncoder()
        parts.append(le.fit_transform(df_features[col].astype(str).str.strip()).reshape(-1, 1))
    X = np.hstack(parts).astype(float)

    assert len(X) == len(y)
    if a is not None:
        assert len(a) == len(y)

    # Split (carry `a` through), then scale on TRAIN ONLY.
    if a is not None:
        X_temp, X_test, y_temp, y_test, a_temp, a_test = train_test_split(
            X, y, a, test_size=test_size, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            X_temp, y_temp, a_temp, test_size=val_size / (1 - test_size),
            random_state=seed, stratify=y_temp
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - test_size),
            random_state=seed, stratify=y_temp
        )
        a_train = a_val = a_test = None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "train": CompasDataset(X_train, y_train, a_train),
        "val": CompasDataset(X_val, y_val, a_val),
        "test": CompasDataset(X_test, y_test, a_test),
        "n_features": X.shape[1],
        "n_classes": 2,
        "sensitive_attribute": sensitive_attribute,
    }
