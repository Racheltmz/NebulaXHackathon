# NebulaX PS3 — evolved classical + deep-learning models (`ian-model`)

Models for the four PS3 subsystems (**Door, ACV, Rail corrugation, SHM**), produced by
OpenEvolve-style LLM evolutionary search (`openevolve==0.3.2` driven through
[openevolve-scientist](https://github.com/xs271828/openevolve-scientist), Codex CLI backend, `gpt-5.6-luna`) against a leak-proof evaluator. Two tracks per task:

* **classical** — CPU-only, runs in a network-less Docker sandbox (numpy / scikit-learn / scipy only).
* **deep** — GPU-only deep learning (PyTorch, frozen TimesFM3 backbone + trained head), each fit/predict executed on an NUS SoC SLURM GPU node.

> **Status: snapshot, runs still in progress.** The 0.99 target has **not** been met on the robust metrics for any task.
> Numbers below were taken 2026-09-19; re-generate with `scripts/collect_best.py`.

## Current best (official metric per task)

`combined = 0.5 · cv3 + 0.5 · train/test`. cv5 is reported but not part of the score.

| Task (metric) | Track | combined | 3-fold | 5-fold | train/test |
|---|---|---:|---:|---:|---:|
| **Door** (accuracy) | classical | **0.976** | 0.952 | 0.940 | 1.000 |
| | deep | 0.911 | 0.867 | 0.841 | 0.955 |
| **ACV** (rank-decay) | classical | **0.948** | 0.896 | 0.900 | 1.000 |
| | deep | 0.646 | 0.542 | 0.475 | 0.750 |
| **Rail** (macro-F1) | classical | **0.796** | 0.665 | 0.600 | 0.926 |
| | deep | 0.481 | 0.489 | 0.419 | 0.473 |
| **SHM** (1 − MAPE) | classical | **0.782** | 0.708 | 0.753 | 0.855 |
| | deep | 0.623 | 0.453 | 0.508 | 0.792 |

Classical clearly leads deep on every task so far. The deep tracks are much younger
(~95 search iterations so far vs 400–610 for classical; every deep evaluation costs minutes of GPU/queue time, and many candidates failed on shape/dtype bugs or infrastructure faults), so the gap is partly an exploration gap — but there is no evidence yet that they will catch up.

### Read these numbers with the data sizes in mind

| Task | train | test | notes |
|---|---:|---:|---|
| Door | 83 | 22 | 60 Normal / 23 Abnormal; test 16 / 6 |
| Rail | 217 | 55 | Normal 187 / Side II 19 / **Side I 11**; test 47 / 5 / **3** |
| ACV | 40 | 8 | 5 faulty in train, **1 faulty test case** → `train/test = 1.000` means one case ranked first |
| SHM | 51 | 13 | continuous target |

* The fixed train/test split is very noisy: one Door test sample ≈ 4.5 pts, one Rail Side-I sample swings macro-F1 by several points. **3-/5-fold CV is the more trustworthy signal**, and it is below 0.95 for ACV, Rail, SHM and every deep track; only Door classical reaches ≈ 0.95 (3-fold) / 0.94 (5-fold).
* Rail is limited by 11 Side-I training recordings; ACV by 5 independent faulty cases. Expect 0.99 to be unreachable there without more data.
* The evolutionary search selects on `combined`, which includes the fixed test split, so best-of-N selection bias applies to `train/test` and `combined`.

## Layout

```
README.md
requirements.txt
common/            contract.py (AST public-signature freeze + safety scan), harness.py (Docker sandbox, CV splits), metrics.py (official metrics)
sandbox/           Dockerfile + sandbox_entry.py  (--network none, classical only)
cloud/             cloud_harness.py (SLURM submit/poll/download), gpu_sandbox_entry.py, ps3evolve_gpu.slurm
data/              ps3_prepare.py, ps3_adapter.py  (raw PS3 -> [N,C,T] .npy; the data itself is NOT in this repo)
scripts/           run_track.sh, collect_best.py, supervisor_flat_layout.sh
docs/              phase1_progress.md (earlier TimesFM work), results_summary.json
phase1_timesfm/    earlier TimesFM3 head-training code (from the cluster) — see "Earlier work"
{door,acv,rail,shm}/
    classical.py            best evolved classical model  (Model.fit / Model.predict)
    deep.py                 best evolved deep model
    baseline_{classical,deep}.py   seed programs; also define the frozen public contract
    eval_{classical,deep}.py       evaluator (3-fold, 5-fold, train/test)
    config_*.yaml, problem_*.yaml  OpenEvolve config + task spec / system prompt
    results_{classical,deep}.json  metrics + provenance of the best program
    checkpoints/            top-10 evolved programs per track (jsonl.gz, Git LFS)
```

## Model interface

Every model is a class with a fixed signature; the evaluator (never the candidate) loads data and scores:

```python
class Model:
    def __init__(self): ...
    def fit(self, X, y): ...      # X: list of [C, T] float arrays; y: labels / targets / fault flags
    def predict(self, X): ...     # classes (Door, Rail), values (SHM), or per-item fault score (ACV)
```

ACV `predict` returns a continuous fault score; the metric ranks cars within each case. Public signatures are frozen by the AST contract check; the search may only mutate code inside the `# EVOLVE-BLOCK` (underscore-prefixed helpers are free to change).

## Evaluation protocol & anti-cheating

* Stratified 3-fold and 5-fold CV on the train split, plus the fixed train/test split; ACV uses group-aware folds so one case never spans train and validation.
* Labels of whatever is being scored never reach the candidate: `predict` gets no labels, and the held-out labels stay in the evaluator process.
* AST contract check rejects any change to the public signatures; a static scan flags dangerous imports/calls.
* **Classical:** candidates run in Docker with `--network none`, memory/CPU/PID limits.
* **Deep:** candidates run on the shared NUS SoC cluster, which has normal outbound network access, so this boundary is **weaker** — the guarantee is structural (held-out labels are never uploaded), not a network block.

## Reproduce

```bash
# 1. data (not included): build arrays from the raw PS3 release
python data/ps3_prepare.py --root <raw PS3 dir> --out data/ps3_arrays      # or set PS3_DATA_DIR

# 2. score an evolved model (classical needs Docker; build the sandbox image first)
docker build -t ps3-evolve-sandbox sandbox/
cd door && python -c "import eval_classical as e; print(e.evaluate('classical.py'))"
#   verified: door/classical.py reproduces 0.976 / 0.9519 / 0.9397 / 1.0 in this layout

# 3. keep evolving a track (needs openevolve==0.3.2 + Codex CLI login)
export OPENEVOLVE_SCRIPT=<...>/openevolve-scientist/skill/openevolve-scientist/scripts/codex_evolution.py
scripts/run_track.sh door classical      # resumes from runs/door_classical/... if present (smoke-tested: 1 iteration end to end)

# 4. refresh classical.py / deep.py / results / checkpoints from a run directory
python scripts/collect_best.py --runs-dir <dir containing <task> and <task>_gpu run folders>
```

Deep evaluation needs the cluster set up: `cloud/cloud_harness.py` shells out to a `saad304` CLI wrapper (upload / download / remote command over the NUS VPN), and `cloud/ps3evolve_gpu.slurm` + the `TIMESFM_SRC` constant in `deep.py` point at the author's cluster home (`/home/s/saad304/...`) — change these for another account. `eval_deep.py` was import-checked but not re-run end to end after the layout change (it consumes GPU jobs).

## Checkpoints

There are **no trained weights** to ship: every model retrains inside `fit()`, and the TimesFM3 backbone is downloaded from `google/timesfm-3.0-pytorch`. What exists are OpenEvolve search checkpoints (the evolutionary archive):

| Where | Size | In repo? |
|---|---:|---|
| Full OpenEvolve checkpoints (8 runs, local: `~/projects/ps3-evolve/*/openevolve_output/checkpoints`) | ≈ 585 MB | No — too large, and mostly redundant near-duplicates |
| Top-10 programs per track, embeddings stripped (`*/checkpoints/*.jsonl.gz`) | ≈ 52 KB total | **Yes, via Git LFS** |

## Earlier work (Phase 1: frozen TimesFM3 heads on the cluster)

Before the evolutionary search, a shared-head repurposing of frozen TimesFM3 was built and run on the cluster
(`phase1_timesfm/`, full write-up in `docs/phase1_progress.md`). Headlines:

* Frozen TimesFM3 + linear head, trained across 41 UCR datasets, best long-context variant averaged ≈ 0.807 on four held-out UCR datasets vs ≈ 0.822 for the FlaMinGo reference — **did not beat the reference**.
* PS3 with TimesFM heads alone was weak: Door 0.9375 BAcc, Rail 0.600, ACV 0.500, SHM 0.435 (score).
* Classical fixed-split baselines from that phase (balanced accuracy, not the official metrics): Door 0.9688, Rail 0.9262 (compact engineered SVC), ACV 1.0 (one test case), SHM 0.6061 (ExtraTrees).
* Its 3-/5-fold audit showed the fixed splits were optimistic (Door 0.85, Rail 0.62–0.68, ACV 0.47, SHM 0.70–0.71) — the same gap the evolved models show.
* Negative results worth not repeating: prototype heads, temporal-statistics heads, shared-attention (SDA) heads, TimesFM+raw fusion, Rail augmentation, and Rail ROCKET / XGBoost / KNN / PCA / stacking all failed to beat the compact SVC.

The evolved deep tracks here start from that idea (frozen TimesFM3 + gated-attention MIL head, `baseline_deep.py`) but are free to change the head, training loop and ensembling.
