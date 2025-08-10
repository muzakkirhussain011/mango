from __future__ import annotations
import os
import numpy as np
import pandas as pd

def load_mimic_eicu(cache_dir: str, sensitive: str):
    """
    Expect a local CSV 'data/mimic_eicu.csv' with columns:
      features... , label, sensitive
    This avoids licensing issues. Provide your own extract.
    """
    path = os.path.join(cache_dir, "mimic_eicu.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected {path}. Provide a local CSV.")
    df = pd.read_csv(path)
    if "label" not in df.columns or "sensitive" not in df.columns:
        raise ValueError("CSV must include 'label' and 'sensitive' columns")
    y = df["label"].astype(int).to_numpy()
    a = df["sensitive"].astype(int).to_numpy()
    X = df.drop(columns=["label","sensitive"]).to_numpy().astype(np.float32)
    return X, y, a
