"""Elman / GRU / LSTM regressors with analytic full-sequence BPTT (numba). Whole training set = one ordered sequence of N steps
(6 features per step), exactly as in the released PyTorch ENN code; h0 = 0; test sequence also starts from h0 = 0."""
import numpy as np, numba as nb

@nb.njit(cache=True)
def _sig(x): return 1.0 / (1.0 + np.exp(-x))

# ---------------- Elman -----------------
@nb.njit(cache=True)
def elman_fwd(X, Wx, Wh, b, Wo, bo):
    N = X.shape[0]; H = Wh.shape[0]
    U = np.zeros((N + 1, H)); yhat = np.empty(N)
    for t in range(N):
        for i in range(H):
            a = b[i]
            for j in range(X.shape[1]): a += Wx[i, j] * X[t, j]
            for j in range(H): a += Wh[i, j] * U[t, j]
            U[t + 1, i] = np.tanh(a)
        s = bo[0]
        for i in range(H): s += Wo[i] * U[t + 1, i]
        yhat[t] = s
    return U, yhat

@nb.njit(cache=True)
def elman_grad(X, y, Wx, Wh, b, Wo, bo):
    N = X.shape[0]; H = Wh.shape[0]
    U, yhat = elman_fwd(X, Wx, Wh, b, Wo, bo)
    gWx = np.zeros_like(Wx); gWh = np.zeros_like(Wh); gb = np.zeros_like(b); gWo = np.zeros_like(Wo); gbo = np.zeros(1)
    dnext = np.zeros(H); loss = 0.0
    for t in range(N - 1, -1, -1):
        e = yhat[t] - y[t]; loss += e * e; dy = 2.0 * e / N
        gbo[0] += dy
        du = np.empty(H)
        for i in range(H):
            gWo[i] += dy * U[t + 1, i]
            du[i] = dy * Wo[i] + dnext[i]
        dpre = du * (1.0 - U[t + 1] ** 2)
        dnext = np.zeros(H)
        for i in range(H):
            gb[i] += dpre[i]
            for j in range(X.shape[1]): gWx[i, j] += dpre[i] * X[t, j]
            for j in range(H):
                gWh[i, j] += dpre[i] * U[t, j]
                dnext[j] += Wh[i, j] * dpre[i]
    return loss / N, gWx, gWh, gb, gWo, gbo

# ---------------- GRU (torch gate convention, single bias) -----------------
@nb.njit(cache=True)
def gru_fwd(X, Wx, Wh, b, Wo, bo, keep):
    N = X.shape[0]; H = Wh.shape[1]
    h = np.zeros(H); yhat = np.empty(N)
    Hs = np.zeros((N + 1, H)); R = np.zeros((N, H)); Z = np.zeros((N, H)); Nn = np.zeros((N, H)); HN = np.zeros((N, H))
    for t in range(N):
        for i in range(H):
            ar = b[i]; az = b[H + i]; an = b[2 * H + i]; hn = 0.0
            for j in range(X.shape[1]):
                ar += Wx[i, j] * X[t, j]; az += Wx[H + i, j] * X[t, j]; an += Wx[2 * H + i, j] * X[t, j]
            for j in range(H):
                ar += Wh[i, j] * h[j]; az += Wh[H + i, j] * h[j]; hn += Wh[2 * H + i, j] * h[j]
            r = _sig(ar); z = _sig(az); n = np.tanh(an + r * hn)
            R[t, i] = r; Z[t, i] = z; Nn[t, i] = n; HN[t, i] = hn
            Hs[t + 1, i] = (1.0 - z) * n + z * h[i]
        for i in range(H): h[i] = Hs[t + 1, i]
        s = bo[0]
        for i in range(H): s += Wo[i] * h[i]
        yhat[t] = s
    return Hs, R, Z, Nn, HN, yhat

@nb.njit(cache=True)
def gru_grad(X, y, Wx, Wh, b, Wo, bo):
    N = X.shape[0]; H = Wh.shape[1]
    Hs, R, Z, Nn, HN, yhat = gru_fwd(X, Wx, Wh, b, Wo, bo, True)
    gWx = np.zeros_like(Wx); gWh = np.zeros_like(Wh); gb = np.zeros_like(b); gWo = np.zeros_like(Wo); gbo = np.zeros(1)
    dnext = np.zeros(H); loss = 0.0
    for t in range(N - 1, -1, -1):
        e = yhat[t] - y[t]; loss += e * e; dy = 2.0 * e / N
        gbo[0] += dy
        dh = np.empty(H)
        for i in range(H):
            gWo[i] += dy * Hs[t + 1, i]; dh[i] = dy * Wo[i] + dnext[i]
        dprev = np.zeros(H)
        for i in range(H):
            r = R[t, i]; z = Z[t, i]; n = Nn[t, i]; hp = Hs[t, i]
            dn = dh[i] * (1.0 - z); dz = dh[i] * (hp - n); dprev[i] += dh[i] * z
            dpn = dn * (1.0 - n * n); dr = dpn * HN[t, i]; dhn = dpn * r
            dpr = dr * r * (1.0 - r); dpz = dz * z * (1.0 - z)
            gb[i] += dpr; gb[H + i] += dpz; gb[2 * H + i] += dpn
            for j in range(X.shape[1]):
                gWx[i, j] += dpr * X[t, j]; gWx[H + i, j] += dpz * X[t, j]; gWx[2 * H + i, j] += dpn * X[t, j]
            for j in range(H):
                hj = Hs[t, j]
                gWh[i, j] += dpr * hj; gWh[H + i, j] += dpz * hj; gWh[2 * H + i, j] += dhn * hj
                dprev[j] += Wh[i, j] * dpr + Wh[H + i, j] * dpz + Wh[2 * H + i, j] * dhn
        dnext = dprev
    return loss / N, gWx, gWh, gb, gWo, gbo

# ---------------- LSTM (gate order i,f,g,o; single bias) -----------------
@nb.njit(cache=True)
def lstm_fwd(X, Wx, Wh, b, Wo, bo):
    N = X.shape[0]; H = Wh.shape[1]
    Hs = np.zeros((N + 1, H)); Cs = np.zeros((N + 1, H)); G = np.zeros((N, 4, H)); yhat = np.empty(N)
    for t in range(N):
        for i in range(H):
            a0 = b[i]; a1 = b[H + i]; a2 = b[2 * H + i]; a3 = b[3 * H + i]
            for j in range(X.shape[1]):
                xv = X[t, j]; a0 += Wx[i, j] * xv; a1 += Wx[H + i, j] * xv; a2 += Wx[2 * H + i, j] * xv; a3 += Wx[3 * H + i, j] * xv
            for j in range(H):
                hv = Hs[t, j]; a0 += Wh[i, j] * hv; a1 += Wh[H + i, j] * hv; a2 += Wh[2 * H + i, j] * hv; a3 += Wh[3 * H + i, j] * hv
            ig = _sig(a0); fg = _sig(a1); gg = np.tanh(a2); og = _sig(a3)
            G[t, 0, i] = ig; G[t, 1, i] = fg; G[t, 2, i] = gg; G[t, 3, i] = og
            Cs[t + 1, i] = fg * Cs[t, i] + ig * gg
            Hs[t + 1, i] = og * np.tanh(Cs[t + 1, i])
        s = bo[0]
        for i in range(H): s += Wo[i] * Hs[t + 1, i]
        yhat[t] = s
    return Hs, Cs, G, yhat

@nb.njit(cache=True)
def lstm_grad(X, y, Wx, Wh, b, Wo, bo):
    N = X.shape[0]; H = Wh.shape[1]
    Hs, Cs, G, yhat = lstm_fwd(X, Wx, Wh, b, Wo, bo)
    gWx = np.zeros_like(Wx); gWh = np.zeros_like(Wh); gb = np.zeros_like(b); gWo = np.zeros_like(Wo); gbo = np.zeros(1)
    dhn = np.zeros(H); dcn = np.zeros(H); loss = 0.0
    for t in range(N - 1, -1, -1):
        e = yhat[t] - y[t]; loss += e * e; dy = 2.0 * e / N
        gbo[0] += dy
        dh = np.empty(H)
        for i in range(H):
            gWo[i] += dy * Hs[t + 1, i]; dh[i] = dy * Wo[i] + dhn[i]
        dhprev = np.zeros(H); dcprev = np.zeros(H)
        for i in range(H):
            ig = G[t, 0, i]; fg = G[t, 1, i]; gg = G[t, 2, i]; og = G[t, 3, i]
            tc = np.tanh(Cs[t + 1, i])
            do = dh[i] * tc
            dc = dcn[i] + dh[i] * og * (1.0 - tc * tc)
            di = dc * gg; dg = dc * ig; df = dc * Cs[t, i]; dcprev[i] = dc * fg
            p0 = di * ig * (1.0 - ig); p1 = df * fg * (1.0 - fg); p2 = dg * (1.0 - gg * gg); p3 = do * og * (1.0 - og)
            gb[i] += p0; gb[H + i] += p1; gb[2 * H + i] += p2; gb[3 * H + i] += p3
            for j in range(X.shape[1]):
                xv = X[t, j]
                gWx[i, j] += p0 * xv; gWx[H + i, j] += p1 * xv; gWx[2 * H + i, j] += p2 * xv; gWx[3 * H + i, j] += p3 * xv
            for j in range(H):
                hv = Hs[t, j]
                gWh[i, j] += p0 * hv; gWh[H + i, j] += p1 * hv; gWh[2 * H + i, j] += p2 * hv; gWh[3 * H + i, j] += p3 * hv
                dhprev[j] += Wh[i, j] * p0 + Wh[H + i, j] * p1 + Wh[2 * H + i, j] * p2 + Wh[3 * H + i, j] * p3
        dhn = dhprev; dcn = dcprev
    return loss / N, gWx, gWh, gb, gWo, gbo

GATES = {"elman": 1, "gru": 3, "lstm": 4}

def n_params(kind, H, nin=6):
    g = GATES[kind]; return g * H * nin + g * H * H + g * H + H + 1

def init(kind, H, rng, nin=6):
    g = GATES[kind]
    xl = lambda r, c: np.sqrt(6.0 / (r + c))
    Wx = rng.uniform(-xl(g*H, nin), xl(g*H, nin), (g*H, nin)); Wh = rng.uniform(-xl(g*H, H), xl(g*H, H), (g*H, H))
    Wo = rng.uniform(-xl(1, H), xl(1, H), H)
    return dict(Wx=Wx, Wh=Wh, b=np.zeros(g*H), Wo=Wo, bo=np.zeros(1))

def _grad(kind, X, y, P):
    f = {"elman": elman_grad, "gru": gru_grad, "lstm": lstm_grad}[kind]
    return f(X, y, P['Wx'], P['Wh'], P['b'], P['Wo'], P['bo'])

def predict(kind, X, P):
    if kind == "elman": return elman_fwd(X, P['Wx'], P['Wh'], P['b'], P['Wo'], P['bo'])[1]
    if kind == "gru": return gru_fwd(X, P['Wx'], P['Wh'], P['b'], P['Wo'], P['bo'], True)[5]
    return lstm_fwd(X, P['Wx'], P['Wh'], P['b'], P['Wo'], P['bo'])[3]

def train(kind, H, Xtr, ytr, Xte, seed, epochs=100, lr=0.01, groups_lr=None):
    """Full-batch SGD. groups_lr=(lr_out, lr_ctx_and_bias, lr_in) is used only for the Elman net (maps to DCQGA's eta1,eta2,eta3)."""
    rng = np.random.default_rng(seed); P = init(kind, H, rng); losses = []
    for _ in range(epochs):
        loss, gWx, gWh, gb, gWo, gbo = _grad(kind, Xtr, ytr, P)
        if groups_lr is None: l_out = l_ctx = l_in = lr
        else: l_out, l_ctx, l_in = groups_lr
        P['Wx'] -= l_in * gWx; P['Wh'] -= l_ctx * gWh; P['b'] -= l_ctx * gb; P['Wo'] -= l_out * gWo; P['bo'] -= l_out * gbo
        losses.append(float(loss))
    return predict(kind, Xte, P), losses, P
