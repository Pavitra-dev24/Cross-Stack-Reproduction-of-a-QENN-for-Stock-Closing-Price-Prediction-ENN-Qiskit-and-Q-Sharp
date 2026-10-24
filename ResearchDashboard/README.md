# Research Experiment Dashboard (React)

Interactive dashboard for fair comparative analysis of:
- Classical ENN variants
- QENN-Qiskit
- QENN-QSharp
- Paper-reported QENN baseline

## What is visualized

1. Mean NMSE comparison by dataset and model
2. Standard deviation (run-to-run variability)
3. Relative gap to paper QENN for quantum implementations
4. Runtime profile (representative)
5. Protocol/fairness controls (split policy, normalization scope, seed policy, fallback mode)
6. Dataset-specific ranking table with CV and delta-vs-paper
7. Local-only derived diagnostics:
   - RMSE estimates computed as `sqrt(NMSE)`
   - 95% confidence interval half-widths from available std values (`1.96 * std / sqrt(10)`)
   - NMSE/RMSE heatmap and sortable aggregate comparison table

## Data sources

The dashboard currently embeds values from:
- `StockPredictionQML/results_table.csv`
- `StockPredictionQML/qenn_gap_table.csv`
- `StockPredictionQML/paper_artifacts/enn_std_by_dataset.csv`
- `StockPredictionQML/paper_artifacts/qenn_fast_std_by_dataset.csv`
- `StockPredictionQML/paper_artifacts/table5_runtime_bse_representative.csv`

No external repo result payloads are used.

## Run locally

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```
