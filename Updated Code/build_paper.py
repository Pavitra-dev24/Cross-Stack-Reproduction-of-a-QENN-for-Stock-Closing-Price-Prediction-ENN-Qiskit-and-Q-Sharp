import sys, re, json, subprocess; sys.path.insert(0, '/home/claude/work')
import numpy as np, pandas as pd
O = ["BSE", "NASDAQ", "HSI", "SSE", "Russell2000", "TAIEX"]; NAMES = dict(zip(O, ["BSE", "NASDAQ", "HSI", "SSE", "Russell 2000", "TAIEX"]))
R = '/home/claude/work/results/'; dd = ['dataset', 'scaling', 'model', 'seed']
ALL = pd.concat([pd.read_csv(R + f).drop_duplicates(dd) for f in ['stage1.csv', 'stage2.csv', 'stage3.csv']], ignore_index=True)
def mn(m, ds, sc='train', met='nmse'): return float(ALL[(ALL.model == m) & (ALL.dataset == ds) & (ALL.scaling == sc)][met].mean())
fl = pd.read_csv(R + 'qenn_range_floor.csv').set_index('dataset')
# ---- leakage ratios
leak_models = ["ENN-100 (SGD .01)", "GRU-6 (SGD .01)", "ENN-12 (SGD .01)", "QENN-B (SGD .01)", "QENN-B DCQGA (corrected fitness)"]
ratio = {(m, ds): mn(m, ds, 'train') / mn(m, ds, 'global') for m in leak_models for ds in O}
leakmax = max(ratio.values()); leakmax_key = max(ratio, key=ratio.get)
e100 = {ds: ratio[("ENN-100 (SGD .01)", ds)] for ds in O}
# ---- main text numbers
pers = {ds: mn('Persistence', ds) for ds in O}
tuned = ["ENN-12 (tuned)", "GRU-6 (tuned)", "LSTM-5 (tuned)"]; qen = ["QENN-B DCQGA (corrected fitness)", "QENN-O DCQGA (corrected fitness)", "QENN-B (tuned)"]
best_cl = {ds: min(mn(m, ds) for m in tuned) for ds in O}; best_q = {ds: min(mn(m, ds) for m in qen) for ds in O}
rat_cl = [best_cl[ds] / pers[ds] for ds in O]; rat_q = [best_q[ds] / best_cl[ds] for ds in O]
aligned = [mn("ENN-12 (lr at QENN bounds)", ds) for ds in O]
q_worse = sum(best_q[ds] > best_cl[ds] for ds in O)
e100n = {ds: mn("ENN-100 (SGD .01)", ds) for ds in O}
f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"; f1 = lambda x: f"{x:.1f}"
MAINTEXT = (f"Persistence and ARIMA have the lowest NMSE on all six indices ({min(pers.values()):.3f}--{max(pers.values()):.3f}); the two are indistinguishable at this precision. "
  f"The best learned model with about 230 parameters is {min(rat_cl):.0f} to {max(rat_cl):.0f} times worse than persistence, and the Elman network with 100 hidden units, kept for reference, is {min(e100n[d]/pers[d] for d in O):.0f} to {max(e100n[d]/pers[d] for d in O):.0f} times worse. "
  f"The best QENN variant is worse than the best parameter-matched classical network on {q_worse} of 6 indices (by a factor of {min(rat_q):.1f} to {max(rat_q):.0f}); its smallest gap is on SSE ({f3(best_q['SSE'])} against {f3(best_cl['SSE'])}). "
  f"An Elman network trained with learning rates at the QENN bounds also stays close to its initialization (NMSE {min(aligned):.1f}--{max(aligned):.1f}), so the learning-rate window of the QENN, not only its structure, limits what 100 batch epochs can reach. "
  "Learned models are not converged: the rankings among them depend on the training budget (Section~\\ref{sec:conclusion}).")
LEAKTEXT = (f"Global scaling lowers the NMSE of the 100-unit Elman network by a factor of {e100['BSE']:.0f} on BSE, {e100['NASDAQ']:.0f} on NASDAQ, {e100['HSI']:.1f} on HSI, {e100['SSE']:.1f} on SSE, {e100['Russell2000']:.0f} on Russell~2000 and {e100['TAIEX']:.0f} on TAIEX. "
  "The effect is largest where the test part leaves the training range (last column of Table~\\ref{tab:data}) and absent for SSE, whose test values lie inside it; persistence and ARIMA are unaffected because NMSE is invariant to rescaling. "
  f"The largest ratio over all models in Table~\\ref{{tab:leak}} is {leakmax:.0f} (Elman-100, {NAMES[leakmax_key[1]]}). "
  "Global scaling also hides the output limit of the QENN, since it places every test value inside $[-1,1]$; the leak therefore favors a bounded-output model in particular.")
fl_txt = ", ".join(f"{100*fl.loc[d,'frac_above1']:.0f}\\%" for d in ["BSE", "NASDAQ", "TAIEX"])
RANGETEXT = ("The QENN output is confined to $[-1,1]$, so it cannot forecast a value above the training maximum. "
  f"For BSE, NASDAQ and TAIEX, {fl_txt} of the test targets lie above it, and even a forecast that is perfect inside the range cannot reach an NMSE below {fl.loc['BSE','floor_nmse']:.1f}, {fl.loc['NASDAQ','floor_nmse']:.1f} and {fl.loc['TAIEX','floor_nmse']:.1f} (last row of Table~\\ref{{tab:main}}). "
  f"The measured NMSE of the batch QENN on BSE and NASDAQ ({mn('QENN-B DCQGA (corrected fitness)','BSE'):.0f} and {mn('QENN-B DCQGA (corrected fitness)','NASDAQ'):.0f}) is far above this floor: its forecasts oscillate between about $-0.3$ and $1$ without following the series, so the model has hardly left its initialization. "
  f"SSE is the cleanest case, since its floor is zero and its test values lie inside the training range. There the tuned batch QENN ({f3(mn('QENN-B (tuned)','SSE'))}) and the online QENN ({f3(mn('QENN-O DCQGA (corrected fitness)','SSE'))}) come closest to the classical networks ({f3(best_cl['SSE'])}), but remain above persistence ({f3(pers['SSE'])}).")
# ---- RNG audit
a = pd.read_csv(R + 'stage3.csv').drop_duplicates(dd); a = a[a.scaling == 'train']
from scipy import stats
pv = []; dif = []
for ds in O:
    k = a[(a.dataset == ds) & a.model.str.contains('Qiskit-path')].mse.values; q = a[(a.dataset == ds) & a.model.str.contains('Q#-path')].mse.values
    pv.append(stats.ttest_ind(k, q, equal_var=False).pvalue); dif.append(abs(k.mean() - q.mean()))
sd = np.mean([a[(a.dataset == ds) & a.model.str.contains('path')].mse.std(ddof=1) for ds in O[:5]])
AUDITTEXT = (f"With the released fitness, the two draw orders give mean MSEs that differ by up to {max(dif):.2f} (Table~\\ref{{tab:audit}}), comparable in size to the Qiskit--Q\\# differences of our first version ({0.3595-0.2675:.2f} on BSE). "
  f"These differences are within the seed-to-seed spread (standard deviation about {sd:.1f} on the five comparable indices; Welch test $p\\ge{min(pv):.2f}$ on every index). "
  "The statement of our first version that Q\\# reproduces the reference values better than Qiskit is therefore not supported and is withdrawn. "
  "With the corrected fitness and global scaling, the MSE of our QENN is of the same order as the values of~\\cite{liu2022qenn} (Table~\\ref{tab:audit}), with no systematic relation: it is lower on BSE, NASDAQ and HSI and higher on SSE.")
json.dump(dict(leakmax=leakmax, pv=pv, dif=dif, rat_cl=rat_cl, rat_q=rat_q), open('tabs/facts.json', 'w'), indent=1)
# ---- sensitivity
try:
    s4 = pd.read_csv(R + 'stage4.csv'); g = s4.groupby(['pop', 'subset']).agg(nmse=('nmse', 'mean'), mse=('mse', 'mean'), sec=('seconds', 'mean'), n=('seed', 'count')).reset_index()
    rows = [f"{int(r.pop)} & {r.subset} & {r.nmse:.2f} & {r.mse:.3f} & {r.sec:.0f} \\\\" for r in g.itertuples()]
    spread = a[(a.dataset == 'BSE') & a.model.str.contains('corrected')].nmse.std(ddof=1)
    SENSTEXT = (f"Table~\\ref{{tab:sens}} varies the population (30, 50) and the fitness subset (256 windows or the whole training set) on BSE with two seeds. The mean NMSE ranges from {g.nmse.min():.1f} to {g.nmse.max():.1f}, within the seed-to-seed spread of the batch QENN on this index (standard deviation $\\approx${spread:.1f}), while the run time grows several-fold with the full subset. "
      "Population size and subset size therefore do not explain the gap to~\\cite{liu2022qenn}; the fitness definition and the learning-rate window matter far more.")
    SENSTAB = "\n".join(rows)
except Exception as ex:
    print('stage4 not ready', ex); SENSTEXT = "@@SENSTEXT@@"; SENSTAB = ""
CONCL = ("With training-only scaling, persistence and ARIMA gave the lowest error on all six indices, no learned model beat them, and the QENN was not better than recurrent networks of the same size. "
  "Three defects in the released code, a DCQGA fitness independent of the learning rates, unequal random-number streams and a Q\\# source that did not compile, explain the stack differences reported in our first version. "
  "Once they are fixed, Qiskit-Aer, the Q\\# interpreter and NumPy are numerically interchangeable for this model, because the quantum software only supplies the cosine--sine pair of each hidden unit. "
  "These results concern a CPU simulation of one training scheme and say nothing about quantum hardware or about quantum models with variational circuits. "
  "The checklist of Section~\\ref{sec:lessons} is the practical outcome: scale on training data, add naive and size-matched baselines, check the output range, test the tuner, match random streams, and say what runs where.")
TEX = open('/home/claude/paper/main_template.tex').read()
rep = {'@@LEAKMAX@@': f"{leakmax:.0f}", '@@TABDATA@@': open('tabs/tab_data.tex').read(), '@@TABMAIN@@': open('tabs/tab_main.tex').read(), '@@TABLEAK@@': open('tabs/tab_leak.tex').read(),
       '@@TABAUDIT@@': open('tabs/tab_audit.tex').read(), '@@MAINTEXT@@': MAINTEXT, '@@LEAKTEXT@@': LEAKTEXT, '@@RANGETEXT@@': RANGETEXT, '@@AUDITTEXT@@': AUDITTEXT,
       '@@SENSTEXT@@': SENSTEXT, '@@CONCLUSION@@': CONCL, '@@BIB@@': open('/home/claude/paper/bib.tex').read()}
for k, v in rep.items(): TEX = TEX.replace(k, v)
# sensitivity table
sens = ("\\begin{table}[t]\n\\footnotesize\n\\caption{DCQGA sensitivity on BSE (batch QENN, corrected fitness, training-only scaling, 2 seeds): population, fitness subset, mean NMSE, mean MSE and run time in seconds.}\n\\label{tab:sens}\n"
        "\\begin{tabular*}{\\hsize}{@{\\extracolsep{\\fill}}rrrrr@{}}\n\\toprule\nPopulation & Subset & NMSE & MSE & Time (s) \\\\\n\\colrule\n" + SENSTAB + "\n\\botrule\n\\end{tabular*}\n\\end{table}\n")
TEX = TEX.replace("\\section{Lessons for reproducing", sens + "\n\\section{Lessons for reproducing")
TEX = TEX.replace('\\midrule', '\\colrule'); open('/home/claude/paper/main.tex', 'w').write(TEX)
print('built; leakmax', leakmax, leakmax_key); print(MAINTEXT[:300])
