from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_heart(cache_dir: str, sensitive: str):
    # OpenML "heart" (id 53). Binary target "class"
    ds = fetch_openml(name="heart", version=1, as_frame=True, parser="auto")
    df = ds.frame.dropna(subset=["class"])
    y = (df["class"] == "present").astype(int).to_numpy()
    # choose sensitive attr: "sex" exists as 0/1; else use age bucket
    if "sex" in df.columns and sensitive == "sex":
        a = df["sex"].astype(int).to_numpy()
        X = df.drop(columns=["class", "sex"]).to_numpy()
    else:
        age = df["age"].astype(float)
        a = (age >= age.median()).astype(int).to_numpy()
        X = df.drop(columns=["class", "age"]).to_numpy()
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y.astype(int), a.astype(int)
