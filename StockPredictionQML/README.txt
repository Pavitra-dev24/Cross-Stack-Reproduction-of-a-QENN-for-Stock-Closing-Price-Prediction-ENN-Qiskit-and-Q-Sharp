StockPredictionQML setup notes (Windows 11)

1) Create venv
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install packages
   pip install --upgrade pip
   pip install numpy pandas matplotlib scipy torch qiskit qiskit-aer qsharp azure-quantum

3) Install .NET SDK for Q# (if not already installed)
   winget install Microsoft.DotNet.SDK.8

4) Run smoke test
   python StockPredictionQML\smoke_test.py

5) Run full experiment
   python StockPredictionQML\experiment.py

If Q# interop fails in environment, the code automatically uses a faithful NumPy fallback.
Results and logs mark this mode through model naming and console output.

Gradient note for reproducibility:
The implemented QENN hidden update uses full backpropagation through the tensor-product state and output layer
(`dy_deta_h = 2 * dp_out * (d_alpha @ v_out)`). This is mathematically stricter than the paper's Eq.11-12 local
proxy form based on `p_tilde(eta_tilde_i)`. We keep this explicit because it improves gradient correctness while preserving
the same forward architecture and loss definition.
