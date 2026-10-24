"""Cross-stack check: do Qiskit/Aer and the (real) Q# interpreter produce the same single-qubit amplitudes, and does a QENN trained
with each amplitude provider follow the SAME trajectory when the random-number stream is matched?"""
import sys, time, json
sys.path.insert(0, '/home/claude/work'); sys.path.insert(0, '/home/claude/repo/StockPredictionQML')
import numpy as np, qsharp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import qenn_core_ref as ref
import data as D

SRC = open('/home/claude/work/QENN.qs').read()
qsharp.eval(SRC)
sim = AerSimulator(method='statevector')
cache = {}
def amp_qiskit(theta):
    t = float(np.clip(theta, -10, 10)); key = round(t, 12)
    if key in cache: return cache[key]
    e = np.exp(t); qc = QuantumCircuit(1); qc.initialize([np.cos(e), np.sin(e)], 0); qc.save_statevector()
    sv = np.real(np.asarray(sim.run(qc).result().get_statevector(qc))); cache[key] = sv; return sv
def amp_qsharp(theta):
    return np.asarray(qsharp.eval(f"QENN.GetAmplitudes({float(theta)!r})"), dtype=float)
def amp_numpy(theta):
    t = float(np.clip(theta, -10, 10)); return np.array([np.cos(np.exp(t)), np.sin(np.exp(t))])

rng = np.random.default_rng(0); th = np.concatenate([rng.uniform(-6, 6, 300), [-10, 0, 10, 12, -12]])
a_np = np.array([amp_numpy(t) for t in th]); a_q = np.array([amp_qsharp(t) for t in th]); a_k = np.array([amp_qiskit(t) for t in th])
res = dict(n_theta=len(th), max_abs_diff_qsharp_vs_numpy=float(np.abs(a_q - a_np).max()), max_abs_diff_qiskit_vs_numpy=float(np.abs(a_k - a_np).max()),
           max_abs_diff_qiskit_vs_qsharp=float(np.abs(a_k - a_q).max()))
t0 = time.time(); [amp_qsharp(0.3) for _ in range(200)]; res['qsharp_ms_per_call'] = (time.time() - t0) / 200 * 1000
res['qsharp_runtime'] = f"qsharp {qsharp.__version__ if hasattr(qsharp,'__version__') else ''}"

# end-to-end: short training (batch mode, fixed lrs) with each provider, identical init and identical data
d = D.prepare('BSE'); X, y = d['Xtr'][:60], d['ytr'][:60]; core = ref.QENNCore()
def run(provider, burn):
    r = np.random.default_rng(0)
    for _ in range(burn): r.uniform(-3, 3)
    p = core.init_params(r); L = []
    for e in range(3): l, p = core.train_epoch_manual_with_provider(X, y, p, (1e-3, 1e-3, 1e-3), provider); L.append(l)
    return np.array(L), p.w_in.copy()
L_np, w_np = run(amp_numpy, 0); L_q, w_q = run(amp_qsharp, 0); L_k, w_k = run(amp_qiskit, 0)
L_k8, w_k8 = run(amp_qiskit, 8)   # what the released Qiskit trainer does: 8 extra RNG draws before init
res.update(loss_traj_maxdiff_qsharp_vs_numpy=float(np.abs(L_q - L_np).max()), loss_traj_maxdiff_qiskit_vs_numpy=float(np.abs(L_k - L_np).max()),
           loss_traj_maxdiff_qiskit_vs_qsharp=float(np.abs(L_k - L_q).max()),
           loss_traj_maxdiff_qiskit_burn8_vs_qsharp=float(np.abs(L_k8 - L_q).max()), init_weights_differ_when_burn8=bool(not np.allclose(w_k8, w_q)),
           loss_qsharp=L_q.tolist(), loss_qiskit_burn8=L_k8.tolist())
json.dump(res, open('/home/claude/work/results/xstack.json', 'w'), indent=1); print(json.dumps(res, indent=1))
