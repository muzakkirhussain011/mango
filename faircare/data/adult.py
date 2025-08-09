# faircare/data/adult.py
import io, requests, pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
TRAIN = UCI_BASE + "adult.data"
TEST  = UCI_BASE + "adult.test"

ADULT_COLUMNS = [
 "age","workclass","fnlwgt","education","education-num","marital-status","occupation",
 "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"
]

def _download(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def load_adult(target="income", sensitive="sex"):
    train = pd.read_csv(io.StringIO(_download(TRAIN)), header=None, names=ADULT_COLUMNS, na_values=" ?").dropna()
    test  = pd.read_csv(io.StringIO(_download(TEST)), header=None, names=ADULT_COLUMNS, skiprows=1, na_values=" ?").dropna()
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df[target] = df[target].str.strip().replace({">50K.":">50K", "<=50K.":"<=50K"})
    y = (df[target].str.contains(">50K")).astype(int).values
    s = (df[sensitive].str.strip().map({"Female":0,"Male":1})).astype(int).values
    X_cat = df.select_dtypes(include=["object"]).drop(columns=[target]).fillna("NA")
    X_num = df.select_dtypes(exclude=["object"]).astype("float32")
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat_enc = enc.fit_transform(X_cat)
    scl = StandardScaler()
    X_num_s = scl.fit_transform(X_num).astype("float32")
    X = np.hstack([X_num_s, X_cat_enc]).astype("float32")
    return X, y, s, list(X_num.columns)+list(enc.get_feature_names_out(X_cat.columns))
