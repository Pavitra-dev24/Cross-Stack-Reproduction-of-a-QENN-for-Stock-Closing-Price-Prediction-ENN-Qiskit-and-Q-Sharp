from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from dcqga import DCQGA, DCQGAConfig, LR_BOUNDS
from qenn_core import QENNCore, QENNTrainHistory


@dataclass
class QiskitQENNResult:
    test_nmse: float
    history: QENNTrainHistory


_AER_STATEVECTOR = AerSimulator(method='statevector')


def _build_single_qubit_state(c0: float, c1: float) -> np.ndarray:
    qc = QuantumCircuit(1)
    qc.initialize([c0, c1], 0)
    qc.save_statevector()
    result = _AER_STATEVECTOR.run(qc).result()
    sv = Statevector(result.get_statevector(qc))
    data = np.asarray(sv.data, dtype=np.complex128)
    return np.real_if_close(data).astype(np.float64)


_AMP_CACHE: dict[float, np.ndarray] = {}


def _amplitudes_from_theta(theta: float) -> np.ndarray:
    t = float(np.clip(theta, -10.0, 10.0))
    key = round(t, 10)
    if key in _AMP_CACHE:
        return _AMP_CACHE[key]
    c0 = float(np.cos(np.exp(t)))
    c1 = float(np.sin(np.exp(t)))
    amp = _build_single_qubit_state(c0, c1)
    _AMP_CACHE[key] = amp
    return amp


def _amplitudes_from_theta_fast(theta: float) -> np.ndarray:
    t = float(np.clip(theta, -10.0, 10.0))
    return np.array([np.cos(np.exp(t)), np.sin(np.exp(t))], dtype=np.float64)


class QENNQiskitTrainer:
    def __init__(self, c: float = 0.5):
        self.core = QENNCore(n_i=6, n_h=5, n_o=1, c=c)
        self._validated_equivalence = False

    def _validate_equivalence(self, rng: np.random.Generator, n_checks: int = 8) -> None:
        if self._validated_equivalence:
            return
        for _ in range(n_checks):
            theta = float(rng.uniform(-3.0, 3.0))
            a_qiskit = _amplitudes_from_theta(theta)
            a_fast = _amplitudes_from_theta_fast(theta)
            if not np.allclose(a_qiskit, a_fast, atol=1e-10):
                raise ValueError("Qiskit and analytical amplitudes diverged.")
        self._validated_equivalence = True

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
    ) -> QiskitQENNResult:
        rng = rng or np.random.default_rng(0)
        self._validate_equivalence(rng)
        params = self.core.init_params(rng)
        losses: List[float] = []
        lr_hist: List[Tuple[float, float, float]] = []

        dcqga = DCQGA(
            DCQGAConfig(
                max_iters=100,
                population_size=dcqga_population_size,
                convergence_window=dcqga_convergence_window,
                rel_tol=dcqga_rel_tol,
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
                    x_train[:n_fit], y_train[:n_fit], ptmp, (l1, l2, l3), _amplitudes_from_theta_fast
                )
                return -nmse

            best_lrs, _ = dcqga.optimize(fit_func, force_iters=dcqga_iters_per_epoch)
            lrs = tuple(float(np.clip(best_lrs[i], LR_BOUNDS[i, 0], LR_BOUNDS[i, 1])) for i in range(3))

            nmse, params = self.core.train_epoch_manual_with_provider(
                x_train, y_train, params, lrs, _amplitudes_from_theta_fast
            )
            losses.append(float(nmse))
            lr_hist.append(lrs)

        test_nmse = self.core.evaluate_with_provider(x_test, y_test, params, _amplitudes_from_theta_fast)
        return QiskitQENNResult(test_nmse=test_nmse, history=QENNTrainHistory(losses, lr_hist))
