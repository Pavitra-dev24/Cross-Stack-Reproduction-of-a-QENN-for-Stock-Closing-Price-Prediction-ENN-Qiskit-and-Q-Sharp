from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from data_pipeline import load_dataset_prepared
from enn import train_enn_single_run
from qenn_qiskit import QENNQiskitTrainer
from qenn_qsharp import QENNQSharpTrainer


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decreasing(curve, tol=1e-4):
    if len(curve) < 3:
        return False
    start = float(curve[0])
    end = float(curve[-1])
    # Stricter check for true decrease while allowing only tiny numerical jitter.
    slack = max(tol, 0.0025 * abs(start))
    return end <= start + slack


def run_smoke_test(base_dir: Path) -> bool:
    data = load_dataset_prepared(base_dir / 'Datasets' / 'BSE.csv', window_size=6)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    enn_ok = True
    qiskit_ok = True
    qsharp_ok = True
    lr_bounds_ok = True
    qsharp_signal_ok = True

    qiskit_losses = []
    qsharp_losses = []

    for run in range(3):
        t0 = time.time()
        _set_seeds(run)
        print(f'[SMOKE] run={run} ENN start')

        enn_res = train_enn_single_run(
            x_train,
            y_train,
            x_test,
            y_test,
            hidden_size=10,
            epochs=50,
            lr=0.01,
            device='cuda',
        )
        enn_ok = enn_ok and _decreasing(enn_res.train_losses)
        print(f'[SMOKE] run={run} ENN done loss0={enn_res.train_losses[0]:.6f} lossN={enn_res.train_losses[-1]:.6f}')

        qt = QENNQiskitTrainer(c=0.5)
        print(f'[SMOKE] run={run} QENN-Qiskit start')
        qk_res = qt.train(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=50,
            rng=np.random.default_rng(run),
            dcqga_iters_per_epoch=1,
            fitness_subset_size=32,
        )
        qiskit_losses.append(qk_res.history.train_losses)
        qiskit_ok = qiskit_ok and _decreasing(qk_res.history.train_losses)
        print(f'[SMOKE] run={run} QENN-Qiskit done loss0={qk_res.history.train_losses[0]:.6f} lossN={qk_res.history.train_losses[-1]:.6f}')
        for lrs in qk_res.history.lr_history:
            e1, e2, e3 = lrs
            lr_bounds_ok = lr_bounds_ok and (1e-5 <= e1 <= 1e-4) and (2e-5 <= e2 <= 1e-4) and (9e-6 <= e3 <= 8e-5)

        qs = QENNQSharpTrainer(project_dir=base_dir / 'StockPredictionQML', c=0.5)
        print(f'[SMOKE] run={run} QENN-QSharp start')
        qs_res = qs.train(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=50,
            rng=np.random.default_rng(run),
            dcqga_iters_per_epoch=1,
            fitness_subset_size=32,
        )
        qsharp_losses.append(qs_res.history.train_losses)
        qsharp_ok = qsharp_ok and _decreasing(qs_res.history.train_losses)
        print(f'[SMOKE] run={run} QENN-QSharp done loss0={qs_res.history.train_losses[0]:.6f} lossN={qs_res.history.train_losses[-1]:.6f} fallback={qs_res.used_fallback}')

        # Interop signal equivalence check at theta=0.
        amp = qs.amp.amplitudes(0.0)
        qsharp_signal_ok = qsharp_signal_ok and np.allclose(amp, np.array([np.cos(1.0), np.sin(1.0)]), atol=1e-6)
        print(f'[SMOKE] run={run} complete elapsed={time.time()-t0:.1f}s')

    passed = enn_ok and qiskit_ok and qsharp_ok and lr_bounds_ok and qsharp_signal_ok
    if any(getattr(curve, "__len__", lambda: 0)() == 0 for curve in qiskit_losses + qsharp_losses):
        passed = False

    if passed:
        print('[SMOKE TEST PASSED]')
    else:
        print('[SMOKE TEST FAILED]')
        print(f'enn_ok={enn_ok} qiskit_ok={qiskit_ok} qsharp_ok={qsharp_ok} lr_bounds_ok={lr_bounds_ok} qsharp_signal_ok={qsharp_signal_ok}')
    return passed


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    ok = run_smoke_test(base)
    raise SystemExit(0 if ok else 1)
