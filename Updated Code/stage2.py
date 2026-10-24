"""S2: tuned-budget baselines. Learning rate and epoch budget chosen on a chronological validation split INSIDE the training set
(first 80% of training windows fit, last 20% validate); then retrained on the full training set for 10 seeds. Train-only scaling."""
from run_common import *
LRS = [0.003, 0.01, 0.03, 0.1, 0.3]
EPOCHS = [100, 300, 1000]

def run_classical(kind, H, Xa, ya, Xb, seed, lr, epochs_list):
    rng = np.random.default_rng(seed); P = rnns.init(kind, H, rng); out = {}
    for ep in range(1, max(epochs_list) + 1):
        loss, gWx, gWh, gb, gWo, gbo = rnns._grad(kind, Xa, ya, P)
        if not np.isfinite(loss): break
        P['Wx'] -= lr * gWx; P['Wh'] -= lr * gWh; P['b'] -= lr * gb; P['Wo'] -= lr * gWo; P['bo'] -= lr * gbo
        if ep in epochs_list: out[ep] = rnns.predict(kind, Xb, P)
    return out

def run_qenn_sgd(Xa, ya, Xb, seed, lr, epochs_list):
    d = None
    o = Q.train_qenn({}, seed, epochs=max(epochs_list), mode="sgd", sgd_lr=lr, Xtr=Xa, ytr=ya, Xte=Xb, ckpts=set(epochs_list))
    return o['ckpt']

sel_rows, rows = [], []
for name in DATASETS:
    d = D.prepare(name, matched=True, scaling="train")
    n_fit = int(len(d['Xtr']) * 0.8)
    Xa, ya, Xv, yv = d['Xtr'][:n_fit], d['ytr'][:n_fit], d['Xtr'][n_fit:], d['ytr'][n_fit:]
    for m in ["ENN-12", "GRU-6", "LSTM-5", "QENN-B"]:
        best = (np.inf, None, None)
        for lr in LRS:
            if m == "QENN-B": ck = run_qenn_sgd(Xa, ya, Xv, 0, lr, EPOCHS)
            else: ck = run_classical(*MODELS_RNN[m], Xa, ya, Xv, 0, lr, EPOCHS)
            for ep, pr in ck.items():
                v = float(np.mean((pr - yv) ** 2)) if np.all(np.isfinite(pr)) else np.inf
                if v < best[0]: best = (v, lr, ep)
        sel_rows.append(dict(dataset=name, model=m, val_mse=best[0], lr=best[1], epochs=best[2]))
        _, lr, ep = best
        for seed in range(10):
            if m == "QENN-B": pr = Q.train_qenn(d, seed, epochs=ep, mode="sgd", sgd_lr=lr)['pred']
            else: pr = rnns.train(*MODELS_RNN[m], d['Xtr'], d['ytr'], d['Xte'], seed, epochs=ep, lr=lr)[0]
            mse, nmse = D.metrics(d['yte'], pr)
            rows.append(dict(dataset=name, scaling="train", model=f"{m} (tuned)", seed=seed, mse=mse, nmse=nmse, test_clip_frac=float(np.mean(d['yte'] > 1.0))))
            if name == "BSE" and seed == 0: np.save(f"{RES}/pred_BSE_{m}_tuned.npy", pr)
        print(name, m, "selected", best, flush=True)
    append_rows("stage2.csv", [r for r in rows if r['dataset'] == name]); append_rows("stage2_selection.csv", [r for r in sel_rows if r['dataset'] == name])
