# Comparative Study of Classical ENN and Quantum ENN

Comparative study and full reproduction-extension project for stock closing price prediction using:
- **Classical Elman Neural Network (ENN)** (PyTorch)
- **Quantum Elman Neural Network (QENN)** using **Qiskit/Qiskit-Aer**
- **Quantum Elman Neural Network (QENN)** implemented in **Microsoft Q#** (with faithful NumPy fallback)

This repository reproduces and extends:

> G. Liu and W. Ma, *A quantum artificial neural network for stock closing price prediction*, Information Sciences, vol. 598, pp. 75–85, 2022.  
> DOI: **10.1016/j.ins.2022.03.064**

---

## 1) Research Objective and Contributions

This project performs a rigorous reproduction and extension across six real-world stock-index datasets, adding:

1. Dual quantum software-stack implementations (**Qiskit** and **Q#**).
2. Cross-stack portability analysis.
3. Quantitative gap analysis against paper-reported QENN values.
4. Reproducibility controls (fixed seeds, smoke test, deterministic split policy).
5. Publication-ready manuscript and figures in IEEE format.

---

## 2) Repository Structure

```text
.
├─ Datasets/                      # BSE, NASDAQ, HSI, SSE, Russell2000, TAIEX CSVs
├─ Extracted/                     # Local reference artifacts (not tracked)
├─ StockPredictionQML/
│  ├─ data_pipeline.py            # Loading, close-column inference, normalization, windows, split
│  ├─ enn.py                      # Classical ENN model and training
│  ├─ qenn_core.py                # Shared QENN math + gradients
│  ├─ qenn_qiskit.py              # QENN implementation via Qiskit
│  ├─ qenn_qsharp.py              # QENN via Q# interop (+ NumPy fallback)
│  ├─ dcqga.py                    # DCQGA learning-rate optimizer
│  ├─ smoke_test.py               # Pre-run smoke gate
│  ├─ experiment.py               # Full 6-dataset experiment orchestration
│  ├─ visualize.py                # Plots and charts
│  ├─ results_table.csv           # Main result table (Table 3)
│  ├─ qenn_gap_table.csv          # Gap analysis table (Table 4)
│  ├─ *.png                       # Result charts
│  └─ paper_artifacts/            # Local manuscript artifacts (not tracked)
├─ ResearchDashboard/             # React dashboard for comparative visualization
│  ├─ src/data/experimentData.ts  # Local experiment metrics + protocol controls
│  ├─ src/App.tsx                 # Panels/charts/tables and metric derivations
│  └─ README.md                   # Dashboard-specific usage and data provenance
└─ README.md
```

---

## 3) Data Pipeline

## Datasets
- BSE, NASDAQ, HSI, SSE, Russell2000, TAIEX
- Daily closing prices

## Close-column inference
1. Prioritize columns containing `"close"` (case-insensitive).
2. Otherwise choose the numeric column with highest non-null count.

## Normalization
Global min-max scaling to `[-1, 1]` per dataset:

`y_i = 2 * (x_i - x_min) / (x_max - x_min) - 1`

## Supervised sample construction
Sliding window size `6`:

`(x_k, x_{k+1}, ..., x_{k+5}) -> x_{k+6}`

## Train-test split
- Chronological split, no shuffling
- 80% train / 20% test

---

## 4) Models

## 4.1 Classical ENN (PyTorch)
- `nn.RNN` with `tanh`
- Hidden sizes: `{10, 20, 40, 70, 100}`
- Xavier-uniform initialization (weights), zero biases
- Output layer: linear (`nn.Linear`)
- Optimizer: SGD, `lr = 0.01`
- Epochs: 100

## 4.2 Quantum ENN (QENN)
- Network size: `6 × 5 × 1`
- Hidden qubits: `5` → amplitude state dimension `2^5 = 32`
- Quantum activations:
  - `f_0(x) = cos(e^x)`
  - `f_1(x) = sin(e^x)`
- Context self-connection gain: `c = 0.5`
- Output mapping:
  - `p(η) = sin(e^η)^2`
  - `ŷ = 2 * p(η_out) - 1`

## QENN-Qiskit
- Qiskit-Aer statevector reference path
- Analytical amplitude equivalence checks at tolerance `1e-10`

## QENN-Q#
- Q# operation `QENN.GetAmplitudes(theta: Double)` via Python interop
- Faithful NumPy fallback if Q# runtime is unavailable

---

## 5) DCQGA Learning-Rate Optimization

DCQGA tunes three learning rates per epoch:

    η₁: output layer
    η₂: context-to-hidden + hidden biases
    η₃: input-to-hidden

Bounds:

    1e-5 < η₁ < 1e-4
    2e-5 < η₂ < 1e-4
    9e-6 < η₃ < 8e-5

Extended runtime settings used for full QENN runs:

    population size: 30
    convergence window: 3
    relative tolerance: 1e-4
    fitness subset: <= 256

---

## 6) Reproducibility Controls

- 10-run averaging with seeds `0..9`
- Seed synchronization across NumPy / Torch / CUDA
- Smoke test gate before full run
- Chronological split only (no shuffle)
- Explicit result-table and gap-table outputs

Smoke test validates:
1. All three training pipelines execute.
2. Loss curves decrease over epochs.
3. DCQGA rates stay in bounds.
4. Q# interop/fallback returns amplitude-equivalent signal.

---

## 7) Research Dashboard (React)

`ResearchDashboard/` provides an interactive, local-data-only dashboard for fair comparative analysis across ENN and QENN variants.

Key dashboard surfaces include:
- Mean NMSE and derived RMSE comparisons
- Standard deviation analysis and dedicated std table
- Quantum gap vs paper baseline
- Runtime profile (representative)
- Heatmap/radar trend views
- Sortable comparison tables with CV and confidence intervals
- Fair-comparison protocol controls

Run locally:

```powershell
cd ResearchDashboard
npm install
npm run dev
```

---

## 8) How to Run

## 8.1 Smoke test
```powershell
python StockPredictionQML\smoke_test.py
```

## 8.2 Full experiment
```powershell
python StockPredictionQML\experiment.py
```

Useful environment variables:

```powershell
$env:QENN_SKIP_SMOKE = "1"
$env:QENN_DCQGA_FIT_SUBSET = "256"
$env:QENN_DCQGA_POP_SIZE = "30"
$env:QENN_DCQGA_CONV_WINDOW = "3"
$env:QENN_DCQGA_REL_TOL = "1e-4"
$env:QENN_PARALLEL_WORKERS = "4"
$env:QENN_REUSE_PREV = "1"   # reuse prior QENN rows if intentionally doing ENN-only rerun
```

---

## 10) Current Result Highlights

- ENN variants achieve the lowest NMSE in this simulation regime.
- QENN-Q# is generally closer to paper-reported QENN values than QENN-Qiskit.
- Qiskit and Q# match at amplitude level pre-training, but diverge over long optimization due to implementation-level numerical/stochastic execution-path effects.

---

## 11) Limitations

- Quantum paths are simulation/runtime based; no fault-tolerant QPU execution in these runs.
- Q# fallback preserves math but is not physical quantum hardware execution.
- QENN uses a stricter full chain-rule gradient through tensor-product hidden states (documented for reproducibility).

---

## 12) Citation

If you use this repository, please cite the original work:

```bibtex
@article{liu2022qann_stock,
  title={A quantum artificial neural network for stock closing price prediction},
  author={Liu, G. and Ma, W.},
  journal={Information Sciences},
  volume={598},
  pages={75--85},
  year={2022},
  doi={10.1016/j.ins.2022.03.064}
}
```

