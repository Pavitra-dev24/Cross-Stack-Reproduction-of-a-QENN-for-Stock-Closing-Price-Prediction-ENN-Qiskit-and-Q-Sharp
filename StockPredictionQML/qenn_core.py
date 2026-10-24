from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


def clamp_for_quantum_activation(x: np.ndarray | float, low: float = -10.0, high: float = 10.0):
    # Clamp before exp to avoid overflow while preserving the exact activation form in the valid range.
    return np.clip(x, low, high)


def f0(x):
    x = clamp_for_quantum_activation(x)
    return np.cos(np.exp(x))


def f1(x):
    x = clamp_for_quantum_activation(x)
    return np.sin(np.exp(x))


def prob_and_derivative(eta):
    # p(eta)=sin(exp(eta))^2, and dp/deta = sin(2*exp(eta))*exp(eta)
    eta_c = clamp_for_quantum_activation(eta)
    eeta = np.exp(eta_c)
    p = np.sin(eeta) ** 2
    dp = np.sin(2.0 * eeta) * eeta
    return p, dp


def tensor_product_hidden_states(states: List[np.ndarray]) -> np.ndarray:
    alpha = np.array([1.0], dtype=np.float64)
    for st in states:
        alpha = np.kron(alpha, st)
    norm = np.linalg.norm(alpha)
    return alpha / (norm + 1e-12)


def initial_context(n_h: int) -> np.ndarray:
    dim = 2 ** n_h
    vec = np.ones(dim, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def update_context(prev_context: np.ndarray, prev_alpha: np.ndarray, c: float = 0.5) -> np.ndarray:
    mixed = c * prev_context + prev_alpha
    return mixed / (np.linalg.norm(mixed) + 1e-12)


@dataclass
class QENNParams:
    w_in: np.ndarray
    w_ctx: np.ndarray
    b_h: np.ndarray
    v_out: np.ndarray
    b_out: float


@dataclass
class QENNTrainHistory:
    train_losses: List[float]
    lr_history: List[Tuple[float, float, float]]


class QENNCore:
    def __init__(self, n_i: int = 6, n_h: int = 5, n_o: int = 1, c: float = 0.5):
        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o
        self.dim = 2 ** n_h
        # c=0.5 is used as a symmetric mid-range feedback gain in (0,1) since the paper does not fix c.
        self.c = c

    def init_params(self, rng: np.random.Generator) -> QENNParams:
        limit_in = np.sqrt(6.0 / (self.n_i + self.n_h))
        limit_ctx = np.sqrt(6.0 / (self.dim + self.n_h))
        limit_out = np.sqrt(6.0 / (self.dim + self.n_o))
        w_in = rng.uniform(-limit_in, limit_in, size=(self.n_h, self.n_i))
        w_ctx = rng.uniform(-limit_ctx, limit_ctx, size=(self.n_h, self.dim))
        b_h = np.zeros(self.n_h, dtype=np.float64)
        v_out = rng.uniform(-limit_out, limit_out, size=(self.dim,))
        b_out = 0.0
        return QENNParams(w_in=w_in, w_ctx=w_ctx, b_h=b_h, v_out=v_out, b_out=b_out)

    def hidden_qubit_states(self, xk: np.ndarray, context: np.ndarray, params: QENNParams) -> Tuple[List[np.ndarray], np.ndarray]:
        etas = params.w_in @ xk + params.b_h + params.w_ctx @ context
        states = []
        for eta in etas:
            states.append(np.array([f0(eta), f1(eta)], dtype=np.float64))
        return states, etas

    def _alpha_and_dalpha(
        self, states: List[np.ndarray], etas: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        alpha = np.array([1.0], dtype=np.float64)
        for st in states:
            alpha = np.kron(alpha, st)
        d_alpha = []
        for i in range(self.n_h):
            st_parts = []
            for j in range(self.n_h):
                if j == i:
                    eta = float(etas[j])
                    eta_c = float(clamp_for_quantum_activation(eta))
                    eeta = float(np.exp(eta_c))
                    st_parts.append(np.array([-np.sin(eeta) * eeta, np.cos(eeta) * eeta], dtype=np.float64))
                else:
                    st_parts.append(states[j])
            d_ai = np.array([1.0], dtype=np.float64)
            for p in st_parts:
                d_ai = np.kron(d_ai, p)
            d_alpha.append(d_ai)
        return alpha, d_alpha

    def hidden_qubit_states_with_provider(
        self,
        xk: np.ndarray,
        context: np.ndarray,
        params: QENNParams,
        amp_provider: Callable[[float], np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        etas = params.w_in @ xk + params.b_h + params.w_ctx @ context
        states = []
        for eta in etas:
            amp = np.asarray(amp_provider(float(eta)), dtype=np.float64)
            norm = np.linalg.norm(amp) + 1e-12
            states.append(amp / norm)
        return states, etas

    def forward_single(self, xk: np.ndarray, context: np.ndarray, params: QENNParams) -> Dict[str, np.ndarray | float]:
        states, eta_h = self.hidden_qubit_states(xk, context, params)
        alpha, d_alpha = self._alpha_and_dalpha(states, eta_h)
        eta_out = float(alpha @ params.v_out + params.b_out)
        p, dp_out = prob_and_derivative(eta_out)
        y_pred = 2.0 * p - 1.0
        # dy/deta_out = 2*dp_out
        dy_deta_out = 2.0 * float(dp_out)
        # d eta_out / d eta_h_i = (d alpha_i / d eta_h_i) dot v
        dy_deta_h = np.array([dy_deta_out * float(d_alpha[i] @ params.v_out) for i in range(self.n_h)], dtype=np.float64)

        return {
            'alpha': alpha,
            'eta_h': eta_h,
            'eta_out': eta_out,
            'dp_out': float(dp_out),
            'dy_deta_h': dy_deta_h,
            'y_pred': float(y_pred),
            'p': float(p),
        }

    def forward_single_with_provider(
        self,
        xk: np.ndarray,
        context: np.ndarray,
        params: QENNParams,
        amp_provider: Callable[[float], np.ndarray],
    ) -> Dict[str, np.ndarray | float]:
        states, eta_h = self.hidden_qubit_states_with_provider(xk, context, params, amp_provider)
        alpha, d_alpha = self._alpha_and_dalpha(states, eta_h)
        eta_out = float(alpha @ params.v_out + params.b_out)
        p, dp_out = prob_and_derivative(eta_out)
        y_pred = 2.0 * p - 1.0

        dy_deta_out = 2.0 * float(dp_out)
        dy_deta_h = np.array([dy_deta_out * float(d_alpha[i] @ params.v_out) for i in range(self.n_h)], dtype=np.float64)
        return {
            'alpha': alpha,
            'eta_h': eta_h,
            'eta_out': eta_out,
            'dp_out': float(dp_out),
            'dy_deta_h': dy_deta_h,
            'y_pred': float(y_pred),
            'p': float(p),
        }

    def train_epoch_manual(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        params: QENNParams,
        lrs: Tuple[float, float, float],
    ) -> Tuple[float, QENNParams]:
        eta1, eta2, eta3 = lrs
        n = len(x_train)

        context = initial_context(self.n_h)
        prev_alpha = initial_context(self.n_h)

        grad_v = np.zeros_like(params.v_out)
        grad_b_out = 0.0
        grad_w_ctx = np.zeros_like(params.w_ctx)
        grad_b_h = np.zeros_like(params.b_h)
        grad_w_in = np.zeros_like(params.w_in)

        loss_acc = 0.0

        for k in range(n):
            if k > 0:
                context = update_context(context, prev_alpha, c=self.c)

            out = self.forward_single(x_train[k], context, params)
            alpha = out['alpha']
            dp_out = out['dp_out']
            dy_deta_h = out['dy_deta_h']
            y_pred = out['y_pred']
            err = y_pred - float(y_train[k])
            loss_acc += err * err

            common = (2.0 / n) * err
            grad_v += common * (2.0 * dp_out) * alpha
            grad_b_out += common * (2.0 * dp_out)

            for i in range(self.n_h):
                grad_w_ctx[i, :] += common * dy_deta_h[i] * context
                grad_b_h[i] += common * dy_deta_h[i]
                grad_w_in[i, :] += common * dy_deta_h[i] * x_train[k]

            prev_alpha = alpha

        # Eq.13 update style with Δ terms from Eq.10-12.
        params.v_out = params.v_out - eta1 * grad_v
        params.b_out = params.b_out - eta1 * grad_b_out
        params.w_ctx = params.w_ctx - eta2 * grad_w_ctx
        params.b_h = params.b_h - eta2 * grad_b_h
        params.w_in = params.w_in - eta3 * grad_w_in

        nmse = loss_acc / n
        return float(nmse), params

    def train_epoch_manual_with_provider(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        params: QENNParams,
        lrs: Tuple[float, float, float],
        amp_provider: Callable[[float], np.ndarray],
    ) -> Tuple[float, QENNParams]:
        eta1, eta2, eta3 = lrs
        n = len(x_train)

        context = initial_context(self.n_h)
        prev_alpha = initial_context(self.n_h)

        grad_v = np.zeros_like(params.v_out)
        grad_b_out = 0.0
        grad_w_ctx = np.zeros_like(params.w_ctx)
        grad_b_h = np.zeros_like(params.b_h)
        grad_w_in = np.zeros_like(params.w_in)
        loss_acc = 0.0

        for k in range(n):
            if k > 0:
                context = update_context(context, prev_alpha, c=self.c)
            out = self.forward_single_with_provider(x_train[k], context, params, amp_provider)
            alpha = out['alpha']
            dp_out = out['dp_out']
            dy_deta_h = out['dy_deta_h']
            y_pred = out['y_pred']
            err = y_pred - float(y_train[k])
            loss_acc += err * err

            common = (2.0 / n) * err
            grad_v += common * (2.0 * dp_out) * alpha
            grad_b_out += common * (2.0 * dp_out)
            for i in range(self.n_h):
                grad_w_ctx[i, :] += common * dy_deta_h[i] * context
                grad_b_h[i] += common * dy_deta_h[i]
                grad_w_in[i, :] += common * dy_deta_h[i] * x_train[k]
            prev_alpha = alpha

        params.v_out = params.v_out - eta1 * grad_v
        params.b_out = params.b_out - eta1 * grad_b_out
        params.w_ctx = params.w_ctx - eta2 * grad_w_ctx
        params.b_h = params.b_h - eta2 * grad_b_h
        params.w_in = params.w_in - eta3 * grad_w_in
        return float(loss_acc / n), params

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, params: QENNParams) -> float:
        context = initial_context(self.n_h)
        prev_alpha = initial_context(self.n_h)
        se = 0.0
        for k in range(len(x_test)):
            if k > 0:
                context = update_context(context, prev_alpha, c=self.c)
            out = self.forward_single(x_test[k], context, params)
            err = out['y_pred'] - float(y_test[k])
            se += err * err
            prev_alpha = out['alpha']
        return float(se / len(x_test))

    def evaluate_with_provider(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        params: QENNParams,
        amp_provider: Callable[[float], np.ndarray],
    ) -> float:
        context = initial_context(self.n_h)
        prev_alpha = initial_context(self.n_h)
        se = 0.0
        for k in range(len(x_test)):
            if k > 0:
                context = update_context(context, prev_alpha, c=self.c)
            out = self.forward_single_with_provider(x_test[k], context, params, amp_provider)
            err = out['y_pred'] - float(y_test[k])
            se += err * err
            prev_alpha = out['alpha']
        return float(se / len(x_test))
