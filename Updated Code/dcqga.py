from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple

import numpy as np


LR_BOUNDS = np.array([
    [1e-5, 1e-4],
    [2e-5, 1e-4],
    [9e-6, 8e-5],
], dtype=np.float64)


@dataclass
class DCQGAConfig:
    population_size: int = 50
    mutation_probability: float = 0.1
    theta0: float = 0.01 * np.pi
    quantum_bits: int = 3
    max_iters: int = 100
    convergence_window: int = 10
    rel_tol: float = 1e-6
    parallel_eval: bool = True
    parallel_grad: bool = True


class DCQGA:
    def __init__(self, config: DCQGAConfig, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self.thetas = self.rng.uniform(0.0, 2.0 * np.pi, size=(config.population_size, config.quantum_bits))
        self.a = np.cos(self.thetas)
        self.b = np.sin(self.thetas)

    def transform(self) -> Tuple[np.ndarray, np.ndarray]:
        low = LR_BOUNDS[:, 0]
        high = LR_BOUNDS[:, 1]
        x_c = 0.5 * (high * (1.0 + self.a) + low * (1.0 - self.a))
        x_s = 0.5 * (high * (1.0 + self.b) + low * (1.0 - self.b))
        return x_c, x_s

    def _numerical_gradient(self, fit_func: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        eps = 1e-7
        g = np.zeros_like(x)
        evals = []
        for j in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[j] = np.clip(xp[j] + eps, LR_BOUNDS[j, 0], LR_BOUNDS[j, 1])
            xm[j] = np.clip(xm[j] - eps, LR_BOUNDS[j, 0], LR_BOUNDS[j, 1])
            evals.append((j, xp, xm))

        def _calc(item):
            j, xp, xm = item
            return j, (fit_func(xp) - fit_func(xm)) / (2.0 * eps)

        if self.cfg.parallel_grad:
            with ThreadPoolExecutor(max_workers=min(6, len(evals))) as ex:
                for j, gj in ex.map(_calc, evals):
                    g[j] = gj
        else:
            for item in evals:
                j, gj = _calc(item)
                g[j] = gj
        return g

    def step(self, fit_func: Callable[[np.ndarray], float]) -> Dict[str, float | np.ndarray]:
        x_c, x_s = self.transform()
        cache: Dict[Tuple[float, ...], float] = {}

        def eval_fit(x: np.ndarray) -> float:
            key = tuple(np.round(x, 15))
            if key in cache:
                return cache[key]
            v = float(fit_func(x))
            cache[key] = v
            return v

        candidates = []
        fits = []
        fit_c = np.zeros(self.cfg.population_size, dtype=np.float64)
        fit_s = np.zeros(self.cfg.population_size, dtype=np.float64)
        eval_items = []
        for i in range(self.cfg.population_size):
            eval_items.append((i, 'c', x_c[i].copy()))
            eval_items.append((i, 's', x_s[i].copy()))

        def _eval(item):
            i, kind, x = item
            return i, kind, x, eval_fit(x)

        if self.cfg.parallel_eval:
            with ThreadPoolExecutor(max_workers=min(12, len(eval_items))) as ex:
                for i, kind, x, fitness in ex.map(_eval, eval_items):
                    candidates.append((i, kind, x, fitness))
                    fits.append(fitness)
                    if kind == 'c':
                        fit_c[i] = fitness
                    else:
                        fit_s[i] = fitness
        else:
            for item in eval_items:
                i, kind, x, fitness = _eval(item)
                candidates.append((i, kind, x, fitness))
                fits.append(fitness)
                if kind == 'c':
                    fit_c[i] = fitness
                else:
                    fit_s[i] = fitness

        best_idx = int(np.argmax(fits))
        bi, bkind, best_x, best_fit = candidates[best_idx]

        # Eq.21 requires numerical gradients of fitness w.r.t learning-rate variables.
        grads = np.array([self._numerical_gradient(eval_fit, x_c[i]) for i in range(self.cfg.population_size)])
        abs_grads = np.abs(grads)
        gmin = float(np.min(abs_grads))
        gmax = float(np.max(abs_grads) + 1e-12)

        a0, b0 = self.a[bi], self.b[bi]
        for i in range(self.cfg.population_size):
            for j in range(self.cfg.quantum_bits):
                A = a0[j] * self.b[i, j] - b0[j] * self.a[i, j]
                direction = np.sign(A) if A != 0 else 1.0
                # Disambiguated Eq.21 in normalized form for stable, bounded rotation.
                denom = max(gmax - gmin, 1e-12)
                normalized = (abs_grads[i, j] - gmin) / denom
                magnitude = self.cfg.theta0 * np.exp(-normalized)
                dtheta = direction * magnitude
                c, s = np.cos(dtheta), np.sin(dtheta)
                a_old, b_old = self.a[i, j], self.b[i, j]
                self.a[i, j] = c * a_old - s * b_old
                self.b[i, j] = s * a_old + c * b_old

                if self.rng.random() < self.cfg.mutation_probability:
                    self.a[i, j], self.b[i, j] = self.b[i, j], self.a[i, j]

        return {'best_fit': float(best_fit), 'best_x': best_x}

    def optimize(
        self,
        fit_func: Callable[[np.ndarray], float],
        force_iters: int | None = None,
    ) -> Tuple[np.ndarray, List[float]]:
        max_iters = self.cfg.max_iters if force_iters is None else force_iters
        best_hist: List[float] = []
        best_x = None

        for t in range(max_iters):
            out = self.step(fit_func)
            best_hist.append(out['best_fit'])
            best_x = out['best_x']

            if t + 1 >= self.cfg.convergence_window:
                window = best_hist[-self.cfg.convergence_window :]
                prev = np.mean(window[:-1])
                curr = window[-1]
                denom = max(abs(prev), 1e-12)
                rel = abs(curr - prev) / denom
                if rel < self.cfg.rel_tol:
                    break

        assert best_x is not None
        return best_x, best_hist
