# faircare/data/heart.py
import io, requests, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

HEART_URL = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"

def load_heart():
    r = requests.get(HEART_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    y = df["target"].astype(int).values
    s = df["sex"].astype(int).values
    X = df.drop(columns=["target"]).values.astype("float32")
    X = StandardScaler().fit_transform(X).astype("float32")
    return X, y, s, df.columns.tolist()
