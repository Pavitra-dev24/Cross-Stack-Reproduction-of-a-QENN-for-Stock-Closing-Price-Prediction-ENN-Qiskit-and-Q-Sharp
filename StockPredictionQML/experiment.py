from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from data_pipeline import DATASET_FILES, load_all_datasets
from enn import train_enn_single_run
from qenn_qiskit import QENNQiskitTrainer
from qenn_qsharp import QENNQSharpTrainer
from smoke_test import run_smoke_test
from visualize import save_comparison_barchart, save_loss_curves, save_lr_evolution


PAPER_QENN = {
    'BSE': 0.23782,
    'NASDAQ': 0.21830,
    'HSI': 0.29263,
    'SSE': 0.22984,
    'Russell2000': 0.25729,
    'TAIEX': 0.10163,
}


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    qenn_rows = df[df['Model'].isin(['QENN-Qiskit', 'QENN-QSharp'])]
    for _, r in qenn_rows.iterrows():
        model = r['Model']
        for ds in DATASET_FILES.keys():
            paper = float(PAPER_QENN[ds])
            val = float(r[ds])
            abs_gap = abs(val - paper)
            rel_gap_pct = (abs_gap / max(abs(paper), 1e-12)) * 100.0
            rows.append(
                {
                    'Model': model,
                    'Dataset': ds,
                    'Paper_QENN': paper,
                    'Reproduced': val,
                    'Abs_Gap': abs_gap,
                    'Rel_Gap_%': rel_gap_pct,
                }
            )
    return pd.DataFrame(rows)


def _run_qenn_pair(
    run: int,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
    out_dir_str: str,
    dcqga_fit_subset: int,
    dcqga_population_size: int,
    dcqga_convergence_window: int,
    dcqga_rel_tol: float,
) -> dict:
    _set_seeds(run)
    qk_trainer = QENNQiskitTrainer(c=0.5)
    qk_res = qk_trainer.train(
        xtr,
        ytr,
        xte,
        yte,
        epochs=100,
        rng=np.random.default_rng(run),
        dcqga_iters_per_epoch=1,
        fitness_subset_size=min(dcqga_fit_subset, len(xtr)),
        dcqga_population_size=dcqga_population_size,
        dcqga_convergence_window=dcqga_convergence_window,
        dcqga_rel_tol=dcqga_rel_tol,
    )

    qs_trainer = QENNQSharpTrainer(project_dir=Path(out_dir_str), c=0.5)
    qs_res = qs_trainer.train(
        xtr,
        ytr,
        xte,
        yte,
        epochs=100,
        rng=np.random.default_rng(run),
        dcqga_iters_per_epoch=1,
        fitness_subset_size=min(dcqga_fit_subset, len(xtr)),
        dcqga_population_size=dcqga_population_size,
        dcqga_convergence_window=dcqga_convergence_window,
        dcqga_rel_tol=dcqga_rel_tol,
    )

    return {
        'run': run,
        'qiskit_nmse': float(qk_res.test_nmse),
        'qsharp_nmse': float(qs_res.test_nmse),
        'qiskit_losses': qk_res.history.train_losses,
        'qsharp_losses': qs_res.history.train_losses,
        'lr_history': qk_res.history.lr_history,
        'qsharp_fallback': bool(qs_res.used_fallback),
    }


def run_full_experiment(base_dir: Path, run_smoke_gate: bool = True) -> None:
    out_dir = base_dir / 'StockPredictionQML'
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_smoke_gate:
        smoke_ok = run_smoke_test(base_dir)
        if not smoke_ok:
            raise RuntimeError('Smoke test failed; aborting full experiment.')
    else:
        print('[SMOKE] skipped by configuration; assuming prior validated smoke pass.')
    os.environ['QENN_QISKIT_FAST_ONLY'] = '0'

    datasets = load_all_datasets(base_dir / 'Datasets', window_size=6)
    dcqga_fit_subset = int(os.environ.get('QENN_DCQGA_FIT_SUBSET', '256'))
    dcqga_population_size = int(os.environ.get('QENN_DCQGA_POP_SIZE', '30'))
    dcqga_convergence_window = int(os.environ.get('QENN_DCQGA_CONV_WINDOW', '3'))
    dcqga_rel_tol = float(os.environ.get('QENN_DCQGA_REL_TOL', '1e-4'))
    qenn_workers = int(os.environ.get('QENN_PARALLEL_WORKERS', '4'))
    reuse_qenn = os.environ.get('QENN_REUSE_PREV', '0') == '1'

    hidden_sizes = [10, 20, 40, 70, 100]
    model_names = [f'ENN, {h} hidden neurons' for h in hidden_sizes] + ['QENN-Qiskit', 'QENN-QSharp']

    results: Dict[str, Dict[str, float]] = {m: {} for m in model_names}

    representative_loss_curves = {}
    representative_lr_history = None

    for dname in DATASET_FILES.keys():
        print(f'[FULL] Dataset={dname} start')
        d = datasets[dname]
        xtr, ytr, xte, yte = d['x_train'], d['y_train'], d['x_test'], d['y_test']

        for h in hidden_sizes:
            run_vals = []
            for run in range(10):
                _set_seeds(run)
                res = train_enn_single_run(xtr, ytr, xte, yte, hidden_size=h, epochs=100, lr=0.01, device='cuda')
                run_vals.append(res.test_nmse)
                if dname == 'BSE' and run == 0 and h == 10:
                    representative_loss_curves['ENN'] = res.train_losses
            results[f'ENN, {h} hidden neurons'][dname] = float(np.mean(run_vals))
            print(f'[FULL] Dataset={dname} ENN-{h} mean_nmse={results[f"ENN, {h} hidden neurons"][dname]:.6f}')

        if reuse_qenn:
            prev_csv = out_dir / 'results_table.csv'
            if not prev_csv.exists():
                raise RuntimeError('QENN_REUSE_PREV=1 requested but results_table.csv not found.')
            prev_df = pd.read_csv(prev_csv)
            results['QENN-Qiskit'][dname] = float(prev_df.loc[prev_df['Model'] == 'QENN-Qiskit', dname].iloc[0])
            results['QENN-QSharp'][dname] = float(prev_df.loc[prev_df['Model'] == 'QENN-QSharp', dname].iloc[0])
            print(f'[FULL] Dataset={dname} QENN reused from prior run')
        else:
            qk_vals = []
            qs_vals = []
            futures = []
            with ProcessPoolExecutor(max_workers=max(1, qenn_workers)) as ex:
                for run in range(10):
                    print(f'[FULL] Dataset={dname} QENN run={run + 1}/10 submitted')
                    futures.append(
                        ex.submit(
                            _run_qenn_pair,
                            run,
                            xtr,
                            ytr,
                            xte,
                            yte,
                            str(out_dir),
                            dcqga_fit_subset,
                            dcqga_population_size,
                            dcqga_convergence_window,
                            dcqga_rel_tol,
                        )
                    )

                run_results = [f.result() for f in as_completed(futures)]
                run_results.sort(key=lambda r: int(r['run']))

            for rr in run_results:
                run = int(rr['run'])
                qk_vals.append(float(rr['qiskit_nmse']))
                qs_vals.append(float(rr['qsharp_nmse']))
                print(
                    f'[FULL] Dataset={dname} QENN run={run + 1}/10 '
                    f'qiskit_nmse={float(rr["qiskit_nmse"]):.6f} qsharp_nmse={float(rr["qsharp_nmse"]):.6f}'
                )
                if dname == 'BSE' and run == 0:
                    representative_loss_curves['QENN-Qiskit'] = rr['qiskit_losses']
                    representative_loss_curves['QENN-QSharp' + (' [Fallback]' if bool(rr['qsharp_fallback']) else '')] = rr['qsharp_losses']
                    representative_lr_history = rr['lr_history']
                if bool(rr['qsharp_fallback']):
                    print('[QENN-QSharp-Fallback] Q# interop unavailable; using faithful NumPy amplitude encoding.')

            results['QENN-Qiskit'][dname] = float(np.mean(qk_vals))
            results['QENN-QSharp'][dname] = float(np.mean(qs_vals))
        print(f'[FULL] Dataset={dname} QENN-Qiskit mean_nmse={results["QENN-Qiskit"][dname]:.6f}')
        print(f'[FULL] Dataset={dname} QENN-QSharp mean_nmse={results["QENN-QSharp"][dname]:.6f}')

    rows = []
    for model in model_names:
        row = {'Model': model}
        row.update(results[model])
        rows.append(row)

    paper_row = {'Model': 'QENN (paper reported)'}
    paper_row.update(PAPER_QENN)
    rows.insert(5, paper_row)

    df = pd.DataFrame(rows)
    col_order = ['Model'] + list(DATASET_FILES.keys())
    df = df[col_order]

    csv_path = out_dir / 'results_table.csv'
    df.to_csv(csv_path, index=False)
    gap_df = _build_gap_table(df)
    gap_csv_path = out_dir / 'qenn_gap_table.csv'
    gap_df.to_csv(gap_csv_path, index=False)

    save_comparison_barchart(df, out_dir / 'comparison_barchart.png')

    if representative_loss_curves:
        save_loss_curves(representative_loss_curves, out_dir / 'loss_curves_BSE.png')
    if representative_lr_history is not None:
        save_lr_evolution(representative_lr_history, out_dir / 'dcqga_lr_evolution.png')

    print('\n=== Table 3 Reproduction + Extensions ===')
    print(df.to_string(index=False))

    print('\n=== Best model per dataset (lowest NMSE) ===')
    compare_df = df[df['Model'] != 'QENN (paper reported)']
    for ds in DATASET_FILES.keys():
        best_idx = compare_df[ds].astype(float).idxmin()
        best_model = compare_df.loc[best_idx, 'Model']
        best_val = float(compare_df.loc[best_idx, ds])
        print(f'{ds}: {best_model} ({best_val:.6f})')

    print('\n=== Consistency vs paper QENN (±10%) ===')
    for model in ['QENN-Qiskit', 'QENN-QSharp']:
        within = []
        for ds in DATASET_FILES.keys():
            v = float(df.loc[df['Model'] == model, ds].iloc[0])
            p = PAPER_QENN[ds]
            within.append(abs(v - p) / p <= 0.10)
        print(f'{model}: {"YES" if all(within) else "NO"}')

    print('\n=== QENN quantitative gap table (vs paper) ===')
    print(gap_df.to_string(index=False))


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    skip_smoke = os.environ.get('QENN_SKIP_SMOKE', '0') == '1'
    run_full_experiment(base, run_smoke_gate=not skip_smoke)
