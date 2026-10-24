"""S1: baselines + aligned-protocol (paper setting: SGD, lr 0.01, 100 epochs) for classical nets and QENN-B/SGD, under both scalings."""
from run_common import *
out = []
preds = {}
done = done_keys('stage1.csv', ['dataset','scaling'])
for scaling in ["train", "global"]:
    for name in DATASETS:
        if (name, scaling) in done: continue
        d = D.prepare(name, matched=True, scaling=scaling)
        rows = []
        def add(model, seed, pr, extra=None):
            mse, nmse = D.metrics(d['yte'], pr)
            r = dict(dataset=name, scaling=scaling, model=model, seed=seed, mse=mse, nmse=nmse, test_clip_frac=float(np.mean(d['yte'] > 1.0)))
            rows.append(r)
        add("Persistence", 0, persistence(d))
        pr, order = arima_forecast(d); add("ARIMA", 0, pr); preds[(name, scaling, 'ARIMA_order')] = order
        for m, (kind, H) in MODELS_RNN.items():
            for seed in range(10):
                pr, _, _ = rnns.train(kind, H, d['Xtr'], d['ytr'], d['Xte'], seed, epochs=100, lr=0.01); add(m + " (SGD .01)", seed, pr)
                if seed == 0 and name == "BSE" and scaling == "train": preds[(m + " (SGD .01)")] = pr
        # Elman-12 trained with learning rates in the DCQGA range (upper bounds), same epochs
        for seed in range(10):
            pr, _, _ = rnns.train("elman", 12, d['Xtr'], d['ytr'], d['Xte'], seed, epochs=100, groups_lr=(1e-4, 1e-4, 8e-5)); add("ENN-12 (lr at QENN bounds)", seed, pr)
        for seed in range(10):
            o = Q.train_qenn(d, seed, epochs=100, mode="sgd", online=False, sgd_lr=0.01); add("QENN-B (SGD .01)", seed, o['pred'])
            if seed == 0 and name == "BSE" and scaling == "train": preds["QENN-B (SGD .01)"] = o['pred']
        append_rows("stage1.csv", rows)
        print(scaling, name, "done", flush=True)
np.save(f"{RES}/stage1_preds.npy", preds, allow_pickle=True)
