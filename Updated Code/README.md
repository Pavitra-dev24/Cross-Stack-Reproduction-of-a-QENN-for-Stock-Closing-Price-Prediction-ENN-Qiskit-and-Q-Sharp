# Corrected QENN audit code (revision)

Run order: `stage1.py` (baselines, aligned protocol, both scalings) -> `stage2.py` (tuned baselines) -> `stage3.py` (DCQGA-trained QENN variants)
-> `stage4.py` (DCQGA sensitivity) -> `xstack.py` (Qiskit-Aer / Q# / NumPy equivalence; needs `pip install qiskit qiskit-aer qsharp`)
-> `make_tables.py`, `figs_results.py`, `build_paper.py`. Paths in the scripts point to /home/claude/...; edit `DATA_DIR`/`RES` in `data.py`/`run_common.py` for your machine.
Everything resumes from the CSV files in `results/`.

Defects found in the originally released code (all fixed here):
1. DCQGA fitness used the pre-update training loss, which does not depend on the learning rates (`qenn_fast.fitness_corrected` fixes it).
2. The Qiskit trainer consumed 8 extra random numbers before initialization, so Qiskit and Q# runs started from different weights (`burn` argument reproduces this).
3. `QENN.qs` called `ExpD`, which does not exist in the current Q# library; the wrapper silently fell back to NumPy. Fixed in `QENN.qs` (`E() ^ x`).
4. Global min-max scaling leaked test-period information (`data.prepare(scaling="train")` is leak-free).
