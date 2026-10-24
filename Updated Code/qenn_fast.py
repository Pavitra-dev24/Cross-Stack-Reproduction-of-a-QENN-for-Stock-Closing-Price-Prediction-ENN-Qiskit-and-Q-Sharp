"""Numba implementation of the QENN of the released repo (same maths as qenn_core.py), with
 (a) a corrected DCQGA fitness (post-update loss) and (b) optional online (pattern-wise) updates."""
import numpy as np, numba as nb
from dcqga import DCQGA, DCQGAConfig, LR_BOUNDS

NH, NI, DIM = 5, 6, 32

@nb.njit(cache=True)
def _fwd(x, ctx, w_in, w_ctx, b_h, v, b_out, alpha, dalpha, deta):
    # returns y_pred, dy/deta_out(=2dp); fills alpha(32), dalpha(5,32), deta[5]
    s = np.empty((NH, 2)); ds = np.empty((NH, 2))
    for i in range(NH):
        eta = b_h[i]
        for j in range(NI): eta += w_in[i, j] * x[j]
        for j in range(DIM): eta += w_ctx[i, j] * ctx[j]
        ec = min(10.0, max(-10.0, eta)); e = np.exp(ec)
        s[i, 0] = np.cos(e); s[i, 1] = np.sin(e)
        ds[i, 0] = -np.sin(e) * e; ds[i, 1] = np.cos(e) * e
    eta_out = b_out
    for idx in range(DIM):
        a = 1.0
        for i in range(NH):
            bit = (idx >> (NH - 1 - i)) & 1
            a *= s[i, bit]
        alpha[idx] = a
        for u in range(NH):
            d = 1.0
            for i in range(NH):
                bit = (idx >> (NH - 1 - i)) & 1
                d *= ds[i, bit] if i == u else s[i, bit]
            dalpha[u, idx] = d
        eta_out += a * v[idx]
    eo = np.exp(min(10.0, max(-10.0, eta_out)))
    p = np.sin(eo) ** 2; dp = np.sin(2.0 * eo) * eo
    for u in range(NH):
        t = 0.0
        for idx in range(DIM): t += dalpha[u, idx] * v[idx]
        deta[u] = 2.0 * dp * t
    return 2.0 * p - 1.0, 2.0 * dp

@nb.njit(cache=True)
def epoch(X, y, n, w_in, w_ctx, b_h, v, b_out, lr1, lr2, lr3, c, online):
    """One pass over the first n windows. batch: one update at the end (mean gradient). online: update after every pattern.
    Returns mean squared error accumulated DURING the pass (pre-update for batch mode), updates parameters in place."""
    ctx = np.ones(DIM) / np.sqrt(DIM); prev = ctx.copy()
    alpha = np.empty(DIM); dalpha = np.empty((NH, DIM)); deta = np.empty(NH)
    g_v = np.zeros(DIM); g_bo = 0.0; g_wc = np.zeros((NH, DIM)); g_bh = np.zeros(NH); g_wi = np.zeros((NH, NI))
    loss = 0.0
    for k in range(n):
        if k > 0:
            nrm = 0.0
            for j in range(DIM):
                ctx[j] = c * ctx[j] + prev[j]; nrm += ctx[j] * ctx[j]
            nrm = np.sqrt(nrm) + 1e-12
            for j in range(DIM): ctx[j] /= nrm
        yp, dy = _fwd(X[k], ctx, w_in, w_ctx, b_h, v, b_out[0], alpha, dalpha, deta)
        err = yp - y[k]; loss += err * err
        common = 2.0 * err if online else (2.0 / n) * err
        if online:
            for j in range(DIM): v[j] -= lr1 * common * dy * alpha[j]
            b_out[0] -= lr1 * common * dy
            for i in range(NH):
                q = common * deta[i]
                for j in range(DIM): w_ctx[i, j] -= lr2 * q * ctx[j]
                b_h[i] -= lr2 * q
                for j in range(NI): w_in[i, j] -= lr3 * q * X[k, j]
        else:
            for j in range(DIM): g_v[j] += common * dy * alpha[j]
            g_bo += common * dy
            for i in range(NH):
                q = common * deta[i]
                for j in range(DIM): g_wc[i, j] += q * ctx[j]
                g_bh[i] += q
                for j in range(NI): g_wi[i, j] += q * X[k, j]
        for j in range(DIM): prev[j] = alpha[j]
    if not online:
        for j in range(DIM): v[j] -= lr1 * g_v[j]
        b_out[0] -= lr1 * g_bo
        for i in range(NH):
            b_h[i] -= lr2 * g_bh[i]
            for j in range(DIM): w_ctx[i, j] -= lr2 * g_wc[i, j]
            for j in range(NI): w_in[i, j] -= lr3 * g_wi[i, j]
    return loss / n

@nb.njit(cache=True)
def predict(X, w_in, w_ctx, b_h, v, b_out, c):
    n = X.shape[0]; out = np.empty(n)
    ctx = np.ones(DIM) / np.sqrt(DIM); prev = ctx.copy()
    alpha = np.empty(DIM); dalpha = np.empty((NH, DIM)); deta = np.empty(NH)
    for k in range(n):
        if k > 0:
            nrm = 0.0
            for j in range(DIM):
                ctx[j] = c * ctx[j] + prev[j]; nrm += ctx[j] * ctx[j]
            nrm = np.sqrt(nrm) + 1e-12
            for j in range(DIM): ctx[j] /= nrm
        yp, dy = _fwd(X[k], ctx, w_in, w_ctx, b_h, v, b_out[0], alpha, dalpha, deta)
        out[k] = yp
        for j in range(DIM): prev[j] = alpha[j]
    return out

def init_params(rng, c=0.5):
    """Same draws/order as the released QENNCore.init_params (Glorot-uniform)."""
    li = np.sqrt(6.0/(NI+NH)); lc = np.sqrt(6.0/(DIM+NH)); lo = np.sqrt(6.0/(DIM+1))
    w_in = rng.uniform(-li, li, (NH, NI)); w_ctx = rng.uniform(-lc, lc, (NH, DIM)); v = rng.uniform(-lo, lo, DIM)
    return dict(w_in=w_in, w_ctx=w_ctx, b_h=np.zeros(NH), v=v, b_out=np.zeros(1))

def n_params(): return NH*NI + NH*DIM + NH + DIM + 1   # 228

def _copy(p): return {k: a.copy() for k, a in p.items()}

def _step(p, X, y, n, lrs, c, online):
    return epoch(X, y, n, p['w_in'], p['w_ctx'], p['b_h'], p['v'], p['b_out'], lrs[0], lrs[1], lrs[2], c, online)

def fitness_corrected(p, X, y, n, lrs, c, online):
    """-MSE on the subset AFTER applying the update implied by lrs (the released code used the pre-update loss, which is independent of lrs)."""
    q = _copy(p); _step(q, X, y, n, lrs, c, online)
    pr = predict(X[:n], q['w_in'], q['w_ctx'], q['b_h'], q['v'], q['b_out'], c)
    return -float(np.mean((pr - y[:n])**2))

def fitness_released(p, X, y, n, lrs, c, online):
    q = _copy(p); return -float(_step(q, X, y, n, lrs, c, online))

def train_qenn(d, seed, epochs=100, mode="dcqga", online=False, pop=50, subset=256, fitness="corrected", sgd_lr=0.01, c=0.5, record=False, burn=0, Xtr=None, ytr=None, Xte=None, ckpts=None):
    """mode: 'dcqga' (three learning rates from DCQGA, one generation per epoch, persistent population) or 'sgd' (fixed lr for all groups)."""
    rng = np.random.default_rng(seed)
    for _ in range(burn): rng.uniform(-3.0, 3.0)   # emulates the 8 RNG draws consumed by the released Qiskit trainer's equivalence check
    p = init_params(rng, c)
    Xtr = d['Xtr'] if Xtr is None else Xtr; ytr = d['ytr'] if ytr is None else ytr; Xte = d['Xte'] if Xte is None else Xte
    n_fit = min(subset, len(Xtr))
    fit_impl = fitness_corrected if fitness == "corrected" else fitness_released
    lr_hist, loss_hist = [], []; ck = {}
    ga = None
    if mode == "dcqga":
        ga = DCQGA(DCQGAConfig(max_iters=100, population_size=pop, convergence_window=10**6, rel_tol=0.0,
                               parallel_eval=False, parallel_grad=False), rng)
    for ep in range(epochs):
        if mode == "dcqga":
            def f(lrs):
                l = [float(np.clip(lrs[i], LR_BOUNDS[i,0], LR_BOUNDS[i,1])) for i in range(3)]
                return fit_impl(p, Xtr, ytr, n_fit, l, c, online)
            best, _ = ga.optimize(f, force_iters=1)
            lrs = tuple(float(np.clip(best[i], LR_BOUNDS[i,0], LR_BOUNDS[i,1])) for i in range(3))
        else:
            lrs = (sgd_lr, sgd_lr, sgd_lr)
        loss = _step(p, Xtr, ytr, len(Xtr), lrs, c, online)
        if record: lr_hist.append(lrs); loss_hist.append(float(loss))
        if ckpts and (ep + 1) in ckpts: ck[ep + 1] = predict(Xte, p['w_in'], p['w_ctx'], p['b_h'], p['v'], p['b_out'], c)
    pred = predict(Xte, p['w_in'], p['w_ctx'], p['b_h'], p['v'], p['b_out'], c)
    out = dict(pred=pred, params=p, ckpt=ck)
    if record: out.update(lr_hist=lr_hist, loss_hist=loss_hist)
    return out
