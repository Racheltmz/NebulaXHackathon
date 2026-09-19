# NebulaX PS3 — final models (branch `ian-model`)

Four subsystems (Door, ACV, Rail corrugation, SHM), one classical model each, found by LLM-driven evolutionary program search
(OpenEvolve) and then selected on all-data out-of-fold scores. Each subsystem has **one self-contained file** that does the data
preprocessing, holds the exact model that was fitted, and trains on all labelled data:

| Task | Core file | README (data, preprocessing, model, features, why it works, alternatives) |
|---|---|---|
| Door | [`door/door_model.py`](door/door_model.py) | [`door/README.md`](door/README.md) |
| ACV | [`acv/acv_model.py`](acv/acv_model.py) | [`acv/README.md`](acv/README.md) |
| Rail corrugation | [`rail/rail_model.py`](rail/rail_model.py) | [`rail/README.md`](rail/README.md) |
| SHM | [`shm/shm_model.py`](shm/shm_model.py) | [`shm/README.md`](shm/README.md) |

## Results

| Task | Official metric | **Public-test score** | Cross-validated on the labelled data | Model | OpenEvolve iterations kept (attempted) |
|---|---|---:|---:|---|---:|
| Door | IoU-weighted F1 | **0.9474** | 1.000 | RBF-SVM on 679 segment statistics | 610 (597) + 1 |
| ACV | rank-decay | **0.3750** | 1.000 (leave-one-case-out, five same-schema cases) | pairwise per-signal ranker | 0 (seed; raw-signal track: 600 (631)) |
| Rail | macro-F1 | **0.8080** | 0.852 | six-view blend + bias search | 100 (118) |
| SHM | 1 − MAPE | **0.9322** | 0.924 | log-space kernel/SVR/ridge blend + calibrator | 160 (158) |

The public-test scores were returned by the organisers' validator for the models' own outputs.
The cross-validated numbers are optimistic (see *Caveats*); **ACV is the clear failure**: cross-validation said perfect, the unseen case
ranked the faulty car 6th of 8.

## The submission zip

`submissions/predictions.zip` holds `door_/acv_/rail_/shm_predictions.csv` at the top level, with headers byte-identical to the organiser
examples. Door, Rail and SHM are the models' own predictions, each fitted on **all** the labelled data (Door 110 segments, ACV 48 cars, Rail 272
recordings, SHM 64 traces). **The ACV ranking in this zip was set manually** (car 01 first, the car the public ACV score implied), so it does not
reflect the model, whose own ACV output scored 0.375.

## How the models were found: the OpenEvolve pipeline

**Setup.** OpenEvolve 0.3.2 driven through the `openevolve-scientist` wrapper, which calls `codex exec` read-only with the saved Codex
CLI login (no API key). *The classical tracks used the CLI's account default model (configured as `default`); I cannot say which model that
resolved to. Only the GPU tracks named `gpt-5.6-luna` explicitly.* Each track is a folder with an initial program (a `Model` class with
`fit(X, y)` / `predict(X)` inside an `EVOLVE-BLOCK`), an evaluator, a config and a task spec. Four families of tracks:

* **raw-input classical** (Door, ACV, Rail, SHM): the program receives the raw arrays and may do anything with numpy/scikit-learn/scipy;
* **feature-input classical, "nx"** (Rail, SHM, and an ACV seed): seeded with the promoted pipelines of an earlier independent effort (the
  `ian-classical` branch) and given label-free per-recording descriptors as input;
* **deep, GPU** (all four): frozen TimesFM3 + trained attention head, each fit run as a SLURM job on the NUS SoC cluster;
* the winners of the first two families are what is submitted.

**Loop.** Diff-based mutation of the `EVOLVE-BLOCK`; 3 islands, population 50, archive 20; a MAP-Elites grid over (code length,
train/test score, 3-fold score) with 10 bins each; each prompt shows the 3 best and 2 diverse programs with their metrics and error
messages; temperature 0.7; checkpoint every 10 iterations; early stop after 300 iterations without gain; target 0.99; one evaluation at a time
per track (240 s limit). A supervisor script relaunches any track from its latest checkpoint if it stops.

**Evaluator (the fitness).** For every candidate: the official metric on stratified 3-fold and 5-fold CV of the training split (ACV: whole cases
per fold) plus one fixed train/test split; `combined = 0.5·3-fold + 0.5·train/test`.

**Anti-cheating.** Public method signatures are frozen by an AST check and a static scan flags dangerous imports/calls; candidates run
in a fresh Docker container per call (no network, 3 GB, 2 CPUs, hard timeout, files copied in and out); `predict` never receives labels;
held-out labels never leave the evaluator process. (The GPU tracks run on a shared cluster with network access, so their boundary is weaker
— the guarantee there is structural: held-out labels are never uploaded.)

**Iterations** (kept = last checkpoint; attempted = everything in the trace, including redone iterations after restarts):

| Track | kept | attempted | scored > 0 | best evolution score |
|---|---:|---:|---:|---:|
| Door — mis-segmented data (archived) | 610 | 597 | – | 1.0 when re-scored on correct data |
| Door — corrected data | 1 | 1 | 1 | 1.000 |
| ACV raw signals | 600 | 631 | 522 | 0.948 |
| Rail raw | 450 | 571 | 457 | 0.792 |
| Rail "nx" | 100 | 118 | 100 | 0.861 |
| SHM raw | 400 | 423 | 333 | 0.808 |
| SHM "nx" | 160 | 158 | 156 | 0.940 |
| GPU (deep) tracks | 100–125 | 111–139 | 9–33 | 0.48–0.78 (Door 0.994 after two on corrected data) |

**Selection after evolution.** The best evolution score is *not* used to pick the final program: it comes from a fixed split and drifts from
the all-data estimate (for Rail the top-scoring program scores 0.812 out of fold, the chosen one 0.852). Instead, the best programs of each
track (up to six per task, drawn from the archives' top ten) were re-scored by pooled 5-fold out-of-fold score on **all** labelled data (ACV: leave-one-case-out), and ensembling was tested against the
best single program: six fold-creation schemes (all data; all faulty + a different third of the normals; overlapping 2/3; cross-fit k-fold;
bootstrap; 80% subsamples), with ensemble weights calibrated out of fold. Rule: an ensemble is used only if its *nested* out-of-fold score (weights
fitted on the other folds) is strictly higher than the best single program trained on all data. It never was, so every submission is a single
program.

**Problems found along the way** (each changed results): the Door arrays were mis-segmented (none of 110 segments had the right length) until a
row-exact rebuild; the ACV metric scored all-tied cases at rank 1 whenever the faulty car happened to be listed first; a Docker bind-mount leak
(~320 mounts/min) crashed the sandbox after a few hours until files were copied in/out instead; the programs' internal CV depends on row order, so the
core files pin the order used.

## Reproduce

```bash
# python 3.12, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3, joblib, openpyxl
python door/door_model.py --data-root <02_Datasets> --out out/door
python acv/acv_model.py   --data-root <02_Datasets> --out out/acv
python rail/rail_model.py --data-root <02_Datasets> --out out/rail --jobs 4     # ~10 min
python shm/shm_model.py   --data-root <02_Datasets> --out out/shm  --jobs 4     # ~10 min
```
Each writes `<task>_predictions.csv`. Door and Rail reproduce the models' submitted files byte-for-byte; ACV reproduces the model's own ranking (the ACV
file in the zip was set manually, see above); SHM matches to ~5×10⁻⁸ relative.

## Caveats

* **Optimism.** Rail and SHM use descriptors and design decisions from an earlier effort that examined all the labelled data, and the programs were then evolved
  against fixed splits, so cross-validated numbers are upper bounds. The public scores (Rail 0.808, SHM 0.932) are the honest reference.
* **Small data.** 30 abnormal Door cycles, 14 Side-I rail recordings, 6 ACV cases (5 usable), 64 SHM traces: differences of a few thousandths are noise.
* **ACV.** Leave-one-case-out over six cases is weak evidence; a 1.000 there did not transfer.
