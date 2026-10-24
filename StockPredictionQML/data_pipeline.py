from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


DATASET_FILES = {
    "BSE": "BSE.csv",
    "NASDAQ": "NASDAQ.csv",
    "HSI": "HSI.csv",
    "SSE": "SSE.csv",
    "Russell2000": "Russell2000.csv",
    "TAIEX": "TAIEX.csv",
}


def infer_close_column(df: pd.DataFrame) -> str:
    numeric_counts = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        numeric_counts[col] = int(s.notna().sum())

    if not numeric_counts:
        raise ValueError("No columns found in CSV.")

    close_name_candidates = [c for c in df.columns if "close" in c.lower()]
    if close_name_candidates:
        best_close = max(close_name_candidates, key=lambda c: numeric_counts[c])
        if numeric_counts[best_close] > 0:
            return best_close

    best_any = max(df.columns, key=lambda c: numeric_counts[c])
    if numeric_counts[best_any] == 0:
        raise ValueError("Unable to infer closing price column (no numeric data).")
    return best_any


def load_closing_prices(csv_path: Path, close_column: Optional[str] = None) -> np.ndarray:
    df = pd.read_csv(csv_path)
    col = close_column if close_column is not None else infer_close_column(df)
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No valid numeric closing prices in column '{col}' ({csv_path}).")
    return series.to_numpy(dtype=np.float64)


def normalize_minus_one_to_one(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_max, x_min):
        return np.zeros_like(x, dtype=np.float64)
    y_min, y_max = -1.0, 1.0
    return (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min


def make_sliding_windows(series: np.ndarray, window_size: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    if len(series) <= window_size:
        raise ValueError("Series too short for chosen window size.")
    x, y = [], []
    for k in range(len(series) - window_size):
        x.append(series[k : k + window_size])
        y.append(series[k + window_size])
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def sequential_train_test_split(
    x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(len(x) * train_ratio)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def load_dataset_prepared(csv_path: Path, window_size: int = 6) -> Dict[str, np.ndarray]:
    raw = load_closing_prices(csv_path)
    norm = normalize_minus_one_to_one(raw)
    x, y = make_sliding_windows(norm, window_size=window_size)
    x_train, y_train, x_test, y_test = sequential_train_test_split(x, y, train_ratio=0.8)
    return {
        "raw": raw,
        "normalized": norm,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def load_all_datasets(dataset_dir: Path, window_size: int = 6) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for name, filename in DATASET_FILES.items():
        out[name] = load_dataset_prepared(dataset_dir / filename, window_size=window_size)
    return out

