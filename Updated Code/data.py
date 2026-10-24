"""Leak-free data pipeline (train-only scaling) with count-matched windows."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

DATA_DIR = Path('C:\\Users\\pavit\\Desktop\\Comparative-Study-of-Classical-ENN-and-Quantum-ENN-main\\Datasets')
FILES = {"BSE":"BSE.csv","NASDAQ":"NASDAQ.csv","HSI":"HSI.csv","SSE":"SSE.csv","Russell2000":"Russell2000.csv","TAIEX":"TAIEX.csv"}
# Sample counts reported in Liu & Ma (2022); TAIEX (1068 samples, 16.7-27.5) cannot be matched -> full series, flagged.
ORIG_COUNT = {"BSE":2454,"NASDAQ":2517,"HSI":2459,"SSE":2859,"Russell2000":2965,"TAIEX":None}
WINDOW = 6

def load_series(name, matched=True):
    df = pd.read_csv(DATA_DIR/FILES[name], skiprows=3, header=None, names=["Date","Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if matched and ORIG_COUNT[name] is not None:
        df = df.iloc[:ORIG_COUNT[name]].reset_index(drop=True)
    return df

def windows(s, w=WINDOW):
    n = len(s) - w
    X = np.stack([s[k:k+w] for k in range(n)])
    y = s[w:w+n].copy()
    return X, y

def prepare(name, matched=True, scaling="train"):
    """scaling='train': min/max from the training portion only (leak-free). 'global': min/max of the whole series (original, leaky)."""
    df = load_series(name, matched)
    raw = df["Close"].to_numpy(np.float64)
    n_win = len(raw) - WINDOW
    n_train = int(n_win * 0.8)
    ref = raw[: n_train + WINDOW] if scaling == "train" else raw   # last training target index = n_train+WINDOW-1
    lo, hi = float(ref.min()), float(ref.max())
    s = 2.0 * (raw - lo) / (hi - lo) - 1.0
    X, y = windows(s)
    d = dict(name=name, dates=df["Date"].to_numpy(), raw=raw, scaled=s, lo=lo, hi=hi,
             Xtr=X[:n_train], ytr=y[:n_train], Xte=X[n_train:], yte=y[n_train:], n_train=n_train)
    d["train_start"] = str(df["Date"].iloc[0].date()); d["train_end"] = str(df["Date"].iloc[n_train+WINDOW-1].date())
    d["test_start"] = str(df["Date"].iloc[n_train+WINDOW].date()); d["test_end"] = str(df["Date"].iloc[-1].date())
    d["n_total"] = len(raw)
    return d

def metrics(y, yhat):
    e = y - yhat
    mse = float(np.mean(e**2))
    nmse = float(np.sum(e**2) / np.sum((y - y.mean())**2))
    return mse, nmse
