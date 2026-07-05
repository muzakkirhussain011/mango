"""UCI Diabetes 130-US hospitals (1999-2008) loader.

Real, freely-downloadable clinical dataset (UCI id 296, ~101k hospital encounters).
Task: predict 30-day readmission. Sensitive attribute: race or gender.

This is the strongest freely-available *healthcare* fairness benchmark in this repo and is
used as the credible replacement for the credentialed MIMIC/eICU datasets.

Reference: Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates:
Analysis of 70,000 Clinical Database Patient Records", BioMed Research International, 2014.
"""
from typing import Optional, Dict
import io
import zipfile
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


# UCI static download (stable). Contains dataset_diabetes/diabetic_data.csv
_UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130+us+hospitals+for+years+1999+2008.zip"
)

# Columns dropped as identifiers or high-missingness (see UCI description / Strack et al.)
_DROP_COLS = [
    "encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty",
]


class DiabetesDataset(Dataset):
    """Diabetes readmission dataset with a binary sensitive attribute."""

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


def _download_diabetes(cache_file: Path) -> pd.DataFrame:
    """Download + extract diabetic_data.csv from the UCI archive to `cache_file`."""
    print("Downloading Diabetes-130 data from UCI (~3 MB zip)...")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(_UCI_ZIP_URL, context=ssl_context) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # The CSV is nested (e.g. dataset_diabetes/diabetic_data.csv); find it by name.
        csv_name = next(n for n in zf.namelist() if n.endswith("diabetic_data.csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    df.to_csv(cache_file, index=False)
    print(f"Cached {len(df)} encounters to {cache_file}")
    return df


def load_diabetes130(
    sensitive_attribute: Optional[str] = "race",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Load the UCI Diabetes 130-US hospitals dataset.

    Args:
        sensitive_attribute: "race" (Caucasian=1 vs other=0), "gender" (Male=1, Female=0), or None.
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
    cache_file = cache_dir / "diabetic_data.csv"

    if cache_file.exists():
        print(f"Loading Diabetes-130 data from cache: {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        df = _download_diabetes(cache_file)

    # Basic cleaning
    df = df.replace("?", np.nan)
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")
    # gender has a handful of 'Unknown/Invalid' rows
    if "gender" in df.columns:
        df = df[df["gender"].isin(["Male", "Female"])]

    # Target: readmitted within 30 days (positive class). '<30' -> 1, {'>30','NO'} -> 0.
    y = (df["readmitted"].astype(str).str.strip() == "<30").astype(int).values

    # Sensitive attribute (extract BEFORE encoding/dropping)
    if sensitive_attribute == "race":
        race = df["race"].astype(str).str.strip()
        a = (race == "Caucasian").astype(int).values  # 1 = Caucasian, 0 = other/unknown
    elif sensitive_attribute in ("gender", "sex"):
        a = (df["gender"].astype(str).str.strip() == "Male").astype(int).values
    else:
        a = None

    # Features = everything except the target. Encode categoricals with LabelEncoder
    # (ordinal ints), fill remaining NaNs, keep numeric columns as-is. Mirrors adult.py.
    df_features = df.drop(columns=["readmitted"])
    numeric_cols, categorical_cols = [], []
    for col in df_features.columns:
        try:
            pd.to_numeric(df_features[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)

    parts = []
    if numeric_cols:
        parts.append(
            df_features[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        )
    for col in categorical_cols:
        le = LabelEncoder()
        enc = le.fit_transform(df_features[col].astype(str).fillna("missing").str.strip())
        parts.append(enc.reshape(-1, 1))
    X = np.hstack(parts).astype(float)

    assert len(X) == len(y)
    if a is not None:
        assert len(a) == len(y)

    # Split (carry `a` through so rows stay aligned), then scale on TRAIN ONLY.
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
        "train": DiabetesDataset(X_train, y_train, a_train),
        "val": DiabetesDataset(X_val, y_val, a_val),
        "test": DiabetesDataset(X_test, y_test, a_test),
        "n_features": X.shape[1],
        "n_classes": 2,
        "sensitive_attribute": sensitive_attribute,
    }
