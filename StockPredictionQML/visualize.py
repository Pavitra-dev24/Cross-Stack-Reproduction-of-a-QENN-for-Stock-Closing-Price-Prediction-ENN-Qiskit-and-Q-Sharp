from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_comparison_barchart(results_wide: pd.DataFrame, out_path: Path) -> None:
    datasets = [c for c in results_wide.columns if c != 'Model']
    models = results_wide['Model'].tolist()
    vals = results_wide[datasets].to_numpy(dtype=float)

    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    plt.figure(figsize=(14, 6))
    for i, model in enumerate(models):
        plt.bar(x + (i - len(models) / 2) * width + width / 2, vals[i], width=width, label=model)

    plt.xticks(x, datasets)
    plt.ylabel('NMSE')
    plt.title('Comparison of Experimental Results (Table 3 + Qiskit + Q#)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_loss_curves(loss_curves: Dict[str, List[float]], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for name, curve in loss_curves.items():
        plt.plot(curve, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training NMSE')
    plt.title('Loss Curves on BSE (Representative Run)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_lr_evolution(lr_history: List[tuple], out_path: Path) -> None:
    arr = np.array(lr_history)
    plt.figure(figsize=(10, 6))
    plt.plot(arr[:, 0], label='eta1')
    plt.plot(arr[:, 1], label='eta2')
    plt.plot(arr[:, 2], label='eta3')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('DCQGA Learning-Rate Evolution (BSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
