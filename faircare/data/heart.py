# faircare/data/heart.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_heart():
    """
    Load the UCI Heart Disease (Cleveland) dataset via the official ucimlrepo API.
    - Converts multiclass 'num' target (0..4) to binary: 1 = disease present (num>0), 0 = none.
    - Coerces all features to numeric, drops rows with missing (e.g., '?' in 'ca'/'thal').
    - Returns standardized float32 features, binary labels, sensitive attr 'sex' (0/1), and feature names.
    """
    try:
        from ucimlrepo import fetch_ucirepo  # official UCI loader
    except Exception as e:
        raise RuntimeError(
            "ucimlrepo is required for loading the UCI Heart dataset. "
            "Install it with: pip install ucimlrepo"
        ) from e

    heart = fetch_ucirepo(id=45)  # UCI 'Heart Disease' (Cleveland) dataset
    X_df = heart.data.features.copy()
    y_df = heart.data.targets.copy()

    # Target: 'num' (0..4). Binarize: >0 => 1 (disease), else 0.
    if "num" not in y_df.columns:
        if "target" in y_df.columns:
            y = y_df["target"].astype(int).values
        else:
            raise RuntimeError("Could not find 'num' or 'target' column in UCI Heart targets.")
    else:
        y = (pd.to_numeric(y_df["num"], errors="coerce").fillna(0) > 0).astype(int).values

    # Coerce features to numeric; UCI files may have '?' strings.
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    # Drop rows with missing values
    mask = ~X_df.isna().any(axis=1)
    X_df = X_df.loc[mask].reset_index(drop=True)
    y = y[mask.values]

    # Sensitive attribute: 'sex' (1 = male, 0 = female)
    if "sex" not in X_df.columns:
        raise RuntimeError("Expected 'sex' in UCI Heart features, but it was not found.")
    s = X_df["sex"].astype(int).values

    # Standardize features
    X = X_df.values.astype("float32")
    X = StandardScaler().fit_transform(X).astype("float32")

    feature_names = list(X_df.columns)
    return X, y, s, feature_names
