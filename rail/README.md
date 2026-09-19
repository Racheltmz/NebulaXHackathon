# Rail corrugation — Normal / Side I / Side II classifier

`rail_model.py` is the single file behind both Rail submissions: raw-data preprocessing (feature extraction from the 128-channel
recordings), the exact OpenEvolve program that was fitted, and a driver that trains on **all** 272 labelled recordings and writes
`rail_predictions.csv`.

| | |
|---|---|
| **Task** | classify each 1-second, 128-channel axle-box recording as `Normal`, `Side I` or `Side II` corrugation |
| **Official metric** | **macro-F1** over the three classes (not accuracy — rare classes count equally) |
| **Public-test score** (no-retraining zip) | **0.8080** macro-F1 |
| **Cross-validated** | 0.852 macro-F1 (5-fold, pooled over all 272 recordings); 0.857 combined score in the evolution protocol (cv3 0.837, cv5 0.788, train/test 0.876) |
| **Model** | six regularised views over 415 label-free descriptors + a tachometer-phase view, with a cross-validated decision-bias search |
| **Evolution** | 100 OpenEvolve iterations kept (118 attempted) on top of a seed program; the program used was found at iteration 84 |

## Data

272 training recordings — **234 Normal, 14 Side I, 24 Side II** — and 68 held-out. Each is a CSV of 10,000 rows (1 s at 10 kHz) and
129 columns: the rotational speed (a toothed wheel with 90 teeth toggling 0/1) followed by vibration and shock for 64 axle boxes
(8 cars × 8 positions, pairs `vibration, shock`). Positions 1,3,5,7 sit on the Side I rail and 2,4,6,8 on Side II; the two rails are
judged separately. The scarcity of Side I (14 recordings) is what limits every result on this task.

## Preprocessing (`rail_vector` and the feature functions)

Every recording is reduced to a 415-vector by label-free, per-recording transforms: **[ v3 (96) | relative (223) | phase (96) ]**.

* **v3 (96).** For each *side* (I, II) and *kind* (vibration, shock), over the 32 sensors of that side/kind: RMS (mean, max), 99th-percentile
  |x| (mean), crest factor, kurtosis and |skewness| (means), spectral-peak ratio and the fraction of energy in the upper 75% of the
  spectrum; the peak frequency (mean and median), peak ratio, and the energy fraction in five bands (0–0.5, 0.5–1, 1–2, 2–4, 4–5 kHz);
  plus mean speed and every *Side I − Side II* difference.
* **relative (223).** v3 without the constant row count (95), plus, for each of the 32 Side-I/Side-II feature pairs, four
  scale-free contrasts: `(a−b)/(|a|+|b|)`, a signed-log ratio, the sum and the max. Corrugation is a one-sided fault, so the
  *contrast between sides* removes the train's overall vibration level.
* **phase (96).** The tachometer transitions give the wheel phase; each channel's |amplitude| is averaged in 64 phase bins (an order
  analysis that ties a fault to a wheel-rotation harmonic); mean/std/peak/90th-percentile of those profiles per side and kind, the first
  harmonics of the Side-I − Side-II profile (cos/sin, absolute and relative), and the estimated speed.
* **Row order** is pinned to a fixed seed-7 order (`LABEL_ORDER`) because the model's internal cross-validation folds depend on it.

## Model

The program blends class-balanced probability views, then adds the phase view:

| view | input | learner |
|---|---|---|
| dense | v3 | scaler → top-50 features (ANOVA F) → logistic (`C=0.03`) |
| dense | relative | top-150 → logistic (`C=0.01`) |
| joint | v3 + relative | top-160 → logistic (`C=0.01`) |
| retrieval | relative | 5 nearest training recordings, distance-weighted class votes |
| hierarchical | relative | fault-vs-normal gate × (Side I vs Side II) classifier |
| sparse | relative | top-80 → RBF-SVM with probabilities |
| phase | phase | top-*k* → logistic, blended in with weight *w* |

The blend weights are 0.12 / 0.35 / 0.15 / 0.10 / 0.13 / 0.15. The final label is `argmax(log p + bias)` with separate additive
biases for Side I and Side II. `k`, `C`, the phase weight *w* and the two biases (a 4×5×6×64 grid) are chosen inside `fit` by two
repeats of stratified inner cross-validation, maximising **macro-F1** (then balanced accuracy).

## Why it works, and why it is the final choice

* **Fold-local selection and strong regularisation.** With 14 Side-I recordings, anything flexible overfits. Every feature-selection
  and scaling step is fitted inside the training fold, heads are logistic with balanced class weights, and several different views are
  averaged so their errors partly cancel.
* **Contrast and phase features** encode what the physics says: corrugation is one-sided and tied to wheel rotation.
* **The bias search targets the actual metric.** Most errors are Normal ↔ fault trades; shifting the Side-I/Side-II decision boundary
  is what macro-F1 rewards, and the inner CV picks it without touching the held-out labels.
* **The metric matters.** The earlier effort on this task reported ~0.92–0.94 *balanced accuracy* for the same kind of model; that is
  ~0.78–0.80 in macro-F1, which is what is scored.
* **Alternatives compared (all-data 5-fold macro-F1):**
  * programs on a compact 5-channel reduction of the raw recordings: 0.69–0.70 — the 128-channel descriptors are what matter;
  * the newest, highest-*evolution*-score program: 0.812, versus 0.852 for the one used — evolution scores are measured on a fixed
    split and drift from the all-data estimate, so the final program was picked on the all-data score;
  * a GPU deep track (frozen TimesFM3 + attention head): 0.48 combined score;
  * **ensembles** (diverse-fold, bootstrap, cross-fit, …): the ensemble's nested out-of-fold score was 0.819 against 0.836 for the
    single program on the same programs, so per the rule (ensemble only if strictly better) the single program is used. The "all faulty +
    a different third of the normals" scheme was the weakest (0.796 vs 0.828).
* **Evolution**: 100 iterations moved the seed's 0.814 combined score to a best of 0.861 (evolution protocol); the program used scores 0.857 there and 0.852 on the all-data estimate.

## Hard-label retraining (the "with" zip)

Pseudo-label only held-out recordings on which at least the cutoff fraction of 10 subsampled copies agree, retrain, predict again.
Tuned on the labelled data with 3-fold CV:

| no retraining | all | **0.6** | 0.7 | 0.8 | 0.9 | 1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8518 | 0.8581 | **0.8637** | 0.8625 | 0.8345 | 0.8291 | 0.8568 |

The curve is not monotonic (0.8 and 0.9 are *worse* than not retraining), so the +0.012 at 0.6 is fragile. On the real test set it
pseudo-labels 66 of 68 recordings and changes one label (`Test13.csv`, Normal → Side I). A separate re-measurement on the final
program found retraining neutral to slightly negative (−0.002).

## What we do not know

The descriptors and design come from an earlier effort that tuned them with cross-validation over all 272 recordings, so 0.852 is an
optimistic estimate; the public score is 0.808. Side I has 14 training examples and 3 in the usual held-out split, so a single
recording moves macro-F1 by several points.

## Reproduce

```bash
python rail_model.py --data-root <02_Datasets> --out out/ --jobs 4     # ~10 min: recomputes the features from the raw files
```
Both output files are byte-identical to the ones in the submitted zips.
