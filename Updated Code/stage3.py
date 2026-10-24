"""S3: DCQGA-trained QENN variants. (a) corrected fitness, batch  (b) corrected fitness, online  (c) RELEASED fitness (pre-update loss), pop=30, with the
Qiskit-path RNG draw order (burn=8) and the Q#-path order (burn=0)  (d) corrected fitness, batch, GLOBAL (leaky) scaling."""
from run_common import *
done = done_keys('stage3.csv', ['dataset', 'model', 'scaling'])
for name in DATASETS:
    for scaling in ["train", "global"]:
        d = D.prepare(name, matched=True, scaling=scaling)
        jobs = []
        if scaling == "train":
            jobs = [("QENN-B DCQGA (corrected fitness)", dict(online=False, pop=50, fitness="corrected", burn=0)),
                    ("QENN-O DCQGA (corrected fitness)", dict(online=True, pop=50, fitness="corrected", burn=0)),
                    ("QENN-B DCQGA (released fitness, Qiskit-path RNG)", dict(online=False, pop=30, fitness="released", burn=8)),
                    ("QENN-B DCQGA (released fitness, Q#-path RNG)", dict(online=False, pop=30, fitness="released", burn=0))]
        else:
            jobs = [("QENN-B DCQGA (corrected fitness)", dict(online=False, pop=50, fitness="corrected", burn=0))]
        for label, kw in jobs:
            if (name, label, scaling) in done: continue
            rows = []
            for seed in range(10):
                o = Q.train_qenn(d, seed, epochs=100, mode="dcqga", subset=256, record=(name == "BSE" and seed == 0 and scaling == "train"), **kw)
                mse, nmse = D.metrics(d['yte'], o['pred'])
                rows.append(dict(dataset=name, scaling=scaling, model=label, seed=seed, mse=mse, nmse=nmse, test_clip_frac=float(np.mean(d['yte'] > 1.0)),
                                 pred_min=float(o['pred'].min()), pred_max=float(o['pred'].max())))
                if name == "BSE" and seed == 0 and scaling == "train":
                    np.save(f"{RES}/pred_BSE_{label.replace(' ', '_').replace('#','S').replace('(','').replace(')','').replace(',','')}.npy", o['pred'])
                    if 'lr_hist' in o: np.save(f"{RES}/lrhist_BSE_{label.split(' ')[0]}_{kw['fitness']}.npy", np.array(o['lr_hist']))
            append_rows("stage3.csv", rows); print(name, scaling, label, "mean nmse", np.mean([r['nmse'] for r in rows]), flush=True)
