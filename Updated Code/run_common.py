import sys, os, json, time, csv
sys.path.insert(0, '/home/claude/work')
import numpy as np
import data as D, rnns, qenn_fast as Q
from baselines import persistence, arima_forecast

RES = 'C:\\Users\\pavit\\Desktop\\Comparative-Study-of-Classical-ENN-and-Quantum-ENN-main\\Results'
MODELS_RNN = {"ENN-12": ("elman", 12), "GRU-6": ("gru", 6), "LSTM-5": ("lstm", 5), "ENN-100": ("elman", 100)}
DATASETS = list(D.FILES)

def append_rows(fname, rows):
    path = f"{RES}/{fname}"; new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)

def done_keys(fname, keycols):
    path = f"{RES}/{fname}"
    if not os.path.exists(path): return set()
    import pandas as pd
    df = pd.read_csv(path); return set(tuple(r) for r in df[keycols].astype(str).values)
