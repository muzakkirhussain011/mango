# faircare/data/mimic_eicu.py
import os, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

def _load_local_csvs(root, features, label, sensitive):
    # Simple reader: expects root to contain X.csv, y.csv, s.csv OR a single data.csv with columns
    x_path = os.path.join(root, "X.csv")
    if os.path.exists(x_path):
        X = pd.read_csv(x_path).values.astype("float32")
        y = pd.read_csv(os.path.join(root,"y.csv")).values.squeeze().astype(int)
        s = pd.read_csv(os.path.join(root,"s.csv")).values.squeeze().astype(int)
        return X, y, s, [f"x{i}" for i in range(X.shape[1])]
    data_path = os.path.join(root, "data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        y = df[label].astype(int).values
        s = df[sensitive].astype(int).values
        X = df[features].values.astype("float32")
        X = StandardScaler().fit_transform(X).astype("float32")
        return X, y, s, features
    raise FileNotFoundError(f"Expected X/y/s CSVs or a single data.csv under {root}")

def load_mimic_demo():
    root = os.getenv("MIMIC_LOCAL_DIR", None)
    if not root:
        raise RuntimeError("Set MIMIC_LOCAL_DIR to a local folder with demo CSVs.")
    # Demo schema expectation: label='mortality', sensitive='sex'
    return _load_local_csvs(root, features=None, label="mortality", sensitive="sex")

def load_eicu_subset():
    root = os.getenv("EICU_LOCAL_DIR", None)
    if not root:
        raise RuntimeError("Set EICU_LOCAL_DIR to a local folder with subset CSVs.")
    return _load_local_csvs(root, features=None, label="mortality", sensitive="sex")
