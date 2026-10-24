import sys, json; sys.path.insert(0, '/home/claude/work')
import numpy as np, pandas as pd
import data as D
ORDER = ["BSE", "NASDAQ", "HSI", "SSE", "Russell2000", "TAIEX"]
HEAD = ["BSE", "NASDAQ", "HSI", "SSE", "Russell 2000", "TAIEX"]
R = '/home/claude/work/results/'
def load(f):
    try: return pd.read_csv(R + f).drop_duplicates([c for c in ['dataset', 'scaling', 'model', 'seed', 'pop', 'subset'] if c in pd.read_csv(R + f).columns])
    except FileNotFoundError: return pd.DataFrame()
s1, s2, s3 = load('stage1.csv'), load('stage2.csv'), load('stage3.csv')
ALL = pd.concat([s1, s2, s3], ignore_index=True)
def f(x, sd=None):
    s = f"{x:.3f}" if x < 10 else (f"{x:.1f}" if x < 1000 else f"{x:.0f}")
    return s
def cell(df, model, ds, scaling='train', metric='nmse', std=True, bold=None):
    q = df[(df.model == model) & (df.dataset == ds) & (df.scaling == scaling)][metric]
    if len(q) == 0: return "--"
    m, s = q.mean(), q.std(ddof=1) if len(q) > 1 else None
    out = f(m) + (f"\\,{{\\tiny$\\pm${f(s)}}}" if (std and s is not None and len(q) > 1) else "")
    return "\\textbf{" + out + "}" if bold else out

# ---------------- Table 1: data periods
rows = []
floor = pd.read_csv(R + 'qenn_range_floor.csv').set_index('dataset')
tick = {"BSE": "\\^{}BSESN", "NASDAQ": "\\^{}IXIC", "HSI": "\\^{}HSI", "SSE": "000001.SS", "Russell2000": "\\^{}RUT", "TAIEX": "\\^{}TWII"}
for n, h in zip(ORDER, HEAD):
    d = D.prepare(n); full = D.load_series(n, matched=False)
    oc = D.ORIG_COUNT[n]
    rows.append(f"{h} & {tick[n]} & {d['train_start']}--{d['train_end']} & {d['test_start']}--{d['test_end']} & {len(d['ytr'])}/{len(d['yte'])} & {oc if oc else '1068$^{\\dagger}$'} & {100*floor.loc[n,'frac_above1']:.0f}\\% \\\\")
open('tabs/tab_data.tex', 'w').write("\n".join(rows))

# ---------------- Table 2: main leak-free results
main_models = [("Persistence", "Persistence"), ("ARIMA", "ARIMA"),
               ("ENN-12 (tuned)", "Elman-12 (tuned)"), ("GRU-6 (tuned)", "GRU-6 (tuned)"), ("LSTM-5 (tuned)", "LSTM-5 (tuned)"),
               ("ENN-100 (SGD .01)", "Elman-100 (SGD, 100 ep.)"), ("ENN-12 (lr at QENN bounds)", "Elman-12, rates at QENN bounds"),
               ("QENN-B DCQGA (corrected fitness)", "QENN-B, DCQGA"), ("QENN-O DCQGA (corrected fitness)", "QENN-O, DCQGA"), ("QENN-B (tuned)", "QENN-B, SGD (tuned)")]
body = []
best = {}
for ds in ORDER:
    vals = {m: ALL[(ALL.model == m) & (ALL.dataset == ds) & (ALL.scaling == 'train')].nmse.mean() for m, _ in main_models}
    vals = {k: v for k, v in vals.items() if not np.isnan(v)}
    mn = min(vals.values()) if vals else None
    best[ds] = [k for k, v in vals.items() if f(v) == f(mn)] if vals else []
for m, lab in main_models:
    body.append(lab + " & " + " & ".join(cell(ALL, m, ds, bold=(m in best[ds])) for ds in ORDER) + " \\\\")
body.append("\\midrule\nQENN output-range floor$^{\\ddagger}$ & " + " & ".join(f(floor.loc[ds, 'floor_nmse']) for ds in ORDER) + " \\\\")
open('tabs/tab_main.tex', 'w').write("\n".join(body))

# ---------------- Table 3: leakage effect
leak_models = [("ENN-100 (SGD .01)", "Elman-100"), ("GRU-6 (SGD .01)", "GRU-6"), ("ENN-12 (SGD .01)", "Elman-12"), ("QENN-B (SGD .01)", "QENN-B, SGD"), ("QENN-B DCQGA (corrected fitness)", "QENN-B, DCQGA")]
body = []
for m, lab in leak_models:
    cells = []
    for ds in ORDER:
        g = ALL[(ALL.model == m) & (ALL.dataset == ds) & (ALL.scaling == 'global')].nmse.mean()
        t = ALL[(ALL.model == m) & (ALL.dataset == ds) & (ALL.scaling == 'train')].nmse.mean()
        cells.append("--" if (np.isnan(g) or np.isnan(t)) else f"{f(g)} / {f(t)}")
    body.append(lab + " & " + " & ".join(cells) + " \\\\")
open('tabs/tab_leak.tex', 'w').write("\n".join(body))

# ---------------- Table 4: audit of the released QENN training (MSE on scaled axis)
PAPER = {"BSE": 0.23782, "NASDAQ": 0.21830, "HSI": 0.29263, "SSE": 0.22984, "Russell2000": 0.25729, "TAIEX": 0.10163}
rows = ["QENN as reported in~\\cite{liu2022qenn} & " + " & ".join(f"{PAPER[ds]:.3f}" if ds != 'TAIEX' else "excl." for ds in ORDER) + " \\\\", "\\midrule"]
for m, sc, lab in [("QENN-B DCQGA (released fitness, Qiskit-path RNG)", 'train', "Released fitness, Qiskit-path RNG, training-only"),
                   ("QENN-B DCQGA (released fitness, Q#-path RNG)", 'train', "Released fitness, Q\\#-path RNG, training-only"),
                   ("QENN-B DCQGA (corrected fitness)", 'train', "Corrected fitness, training-only"),
                   ("QENN-B DCQGA (corrected fitness)", 'global', "Corrected fitness, global scaling")]:
    rows.append(lab + " & " + " & ".join(cell(ALL, m, ds, scaling=sc, metric='mse', std=False) + ("$^{*}$" if ds == 'TAIEX' else "") for ds in ORDER) + " \\\\")
open('tabs/tab_audit.tex', 'w').write("\n".join(rows))

# ---------------- numbers for text
nums = {}
def mean_of(m, ds, sc='train', met='nmse'):
    q = ALL[(ALL.model == m) & (ALL.dataset == ds) & (ALL.scaling == sc)][met]; return float(q.mean()) if len(q) else float('nan')
nums['persist'] = {ds: mean_of('Persistence', ds) for ds in ORDER}
nums['best'] = best
json.dump(nums, open('tabs/nums.json', 'w'), indent=1)
s4 = load('stage4.csv')
if len(s4):
    g = s4.groupby(['pop', 'subset']).agg(nmse=('nmse', 'mean'), mse=('mse', 'mean'), sec=('seconds', 'mean')).reset_index()
    rows = [f"{int(r.pop)} & {r.subset} & {r.nmse:.3f} & {r.mse:.4f} & {r.sec:.0f} \\\\" for r in g.itertuples()]
    open('tabs/tab_sens.tex', 'w').write("\n".join(rows))
print("tables written; best per dataset:", best)
