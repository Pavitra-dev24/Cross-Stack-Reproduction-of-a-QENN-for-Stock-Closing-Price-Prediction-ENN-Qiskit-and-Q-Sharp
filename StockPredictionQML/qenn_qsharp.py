from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from dcqga import DCQGA, DCQGAConfig, LR_BOUNDS
from qenn_core import QENNCore, QENNTrainHistory


@dataclass
class QSharpQENNResult:
    test_nmse: float
    history: QENNTrainHistory
    used_fallback: bool


class QENNAmpProvider:
    def __init__(self, qsharp_project_dir: Path):
        self.used_fallback = False
        self.qsharp = None
        self.qsharp_project_dir = qsharp_project_dir
        try:
            import qsharp  # type: ignore

            self.qsharp = qsharp
            try:
                # Try to load Q# source into runtime if available.
                qsharp.init(project_root=str(qsharp_project_dir))
            except Exception:
                pass
        except Exception:
            self.used_fallback = True

    def amplitudes(self, theta: float) -> np.ndarray:
        val = float(np.clip(theta, -10.0, 10.0))
        if self.qsharp is None:
            self.used_fallback = True
            return np.array([np.cos(np.exp(val)), np.sin(np.exp(val))], dtype=np.float64)

        try:
            # If Q# operation is available, evaluate it. Fallback stays exact on failure.
            # Expected operation name in QENN.qs: QENN.GetAmplitudes
            amps = self.qsharp.eval(f"QENN.GetAmplitudes({val})")
            arr = np.asarray(amps, dtype=np.float64)
            if arr.shape != (2,):
                raise ValueError('Unexpected Q# amplitude shape')
            return arr
        except Exception:
            self.used_fallback = True
            self.qsharp = None
            return np.array([np.cos(np.exp(val)), np.sin(np.exp(val))], dtype=np.float64)


class QENNQSharpTrainer:
    def __init__(self, project_dir: Path, c: float = 0.5):
        self.core = QENNCore(n_i=6, n_h=5, n_o=1, c=c)
        self.amp = QENNAmpProvider(project_dir)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        rng: np.random.Generator | None = None,
        dcqga_iters_per_epoch: int = 1,
        fitness_subset_size: int | None = None,
        dcqga_population_size: int = 50,
        dcqga_convergence_window: int = 10,
        dcqga_rel_tol: float = 1e-6,
    ) -> QSharpQENNResult:
        rng = rng or np.random.default_rng(0)
        params = self.core.init_params(rng)

        losses: List[float] = []
        lr_hist: List[Tuple[float, float, float]] = []

        # Probe once on the main thread so provider can enter deterministic fallback mode if needed.
        _ = self.amp.amplitudes(0.0)
        using_qsharp_runtime = self.amp.qsharp is not None
        # qsharp interpreter is thread-affine; fallback NumPy mode can safely use parallel DCQGA.
        dcqga = DCQGA(
            DCQGAConfig(
                max_iters=100,
                population_size=dcqga_population_size,
                convergence_window=dcqga_convergence_window,
                rel_tol=dcqga_rel_tol,
                parallel_eval=not using_qsharp_runtime,
                parallel_grad=not using_qsharp_runtime,
            ),
            rng,
        )
        n_fit = len(x_train) if fitness_subset_size is None else min(fitness_subset_size, len(x_train))

        for _ in range(epochs):
            def fit_func(lrs: np.ndarray) -> float:
                l1, l2, l3 = [float(v) for v in lrs]
                l1 = float(np.clip(l1, LR_BOUNDS[0, 0], LR_BOUNDS[0, 1]))
                l2 = float(np.clip(l2, LR_BOUNDS[1, 0], LR_BOUNDS[1, 1]))
                l3 = float(np.clip(l3, LR_BOUNDS[2, 0], LR_BOUNDS[2, 1]))
                ptmp = type(params)(
                    w_in=params.w_in.copy(),
                    w_ctx=params.w_ctx.copy(),
                    b_h=params.b_h.copy(),
                    v_out=params.v_out.copy(),
                    b_out=float(params.b_out),
                )
                nmse, _ = self.core.train_epoch_manual_with_provider(
                    x_train[:n_fit], y_train[:n_fit], ptmp, (l1, l2, l3), self.amp.amplitudes
                )
                return -nmse

            best_lrs, _ = dcqga.optimize(fit_func, force_iters=dcqga_iters_per_epoch)
            lrs = tuple(float(np.clip(best_lrs[i], LR_BOUNDS[i, 0], LR_BOUNDS[i, 1])) for i in range(3))
            nmse, params = self.core.train_epoch_manual_with_provider(
                x_train, y_train, params, lrs, self.amp.amplitudes
            )
            losses.append(float(nmse))
            lr_hist.append(lrs)

        test_nmse = self.core.evaluate_with_provider(x_test, y_test, params, self.amp.amplitudes)
        return QSharpQENNResult(
            test_nmse=test_nmse,
            history=QENNTrainHistory(losses, lr_hist),
            used_fallback=self.amp.used_fallback,
        )
