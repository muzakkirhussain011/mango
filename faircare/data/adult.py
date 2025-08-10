from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml

def load_adult(cache_dir: str, sensitive: str):
    """
    Returns X (float), y (int), a (int sensitive group)
    sensitive in {"sex","race"}
    """
    ds = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = ds.frame.dropna(subset=["class"])
    y = (df["class"] == ">50K").astype(int).to_numpy()

    sens_map = {"sex": "sex", "race": "race"}
    sens_col = sens_map.get(sensitive, "sex")
    a = pd.factorize(df[sens_col])[0]

    X_df = df.drop(columns=["class", sens_col])
    cat_cols = X_df.select_dtypes(include=["category","object"]).columns.tolist()
    num_cols = X_df.select_dtypes(include=["number","float","int"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), num_cols),
        ]
    )
    X = pre.fit_transform(X_df).astype(np.float32)
    return X.toarray() if hasattr(X, "toarray") else X, y.astype(int), a.astype(int)
