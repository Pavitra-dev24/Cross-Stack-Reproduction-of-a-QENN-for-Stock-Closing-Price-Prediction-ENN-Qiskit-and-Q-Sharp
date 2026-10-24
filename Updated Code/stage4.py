"""S4: DCQGA sensitivity on BSE (2 seeds): population {30,50} x fitness-subset {256, full training set}; corrected fitness, batch. Saves every run."""
from run_common import *
d = D.prepare("BSE", matched=True, scaling="train")
done = done_keys('stage4.csv', ['pop', 'subset', 'seed'])
for pop in (30, 50):
    for sub in (256, len(d['Xtr'])):
        for seed in range(2):
            lab = "full" if sub == len(d['Xtr']) else str(sub)
            if (str(pop), lab, str(seed)) in done: continue
            t = time.time(); o = Q.train_qenn(d, seed, epochs=100, mode="dcqga", online=False, pop=pop, subset=sub, fitness="corrected")
            mse, nmse = D.metrics(d['yte'], o['pred'])
            append_rows("stage4.csv", [dict(dataset="BSE", pop=pop, subset=lab, seed=seed, mse=mse, nmse=nmse, seconds=time.time() - t)])
            print(pop, lab, seed, round(nmse, 3), flush=True)
