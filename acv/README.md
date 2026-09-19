# ACV — refrigerant-leak car ranking

`acv_model.py` is the single file behind both ACV submissions: raw-data preprocessing, the exact program that was fitted, and a driver
that trains on **all** labelled cars and writes `acv_predictions.csv`.

| | |
|---|---|
| **Task** | each case = 8 cars, exactly one has a refrigerant leak; rank the cars from most to least likely faulty |
| **Official metric** | rank-decay: `(8 − (rank − 1)) / 8` for the true faulty car (rank 1 → 1.0, rank 6 → 0.375) |
| **Public-test score** (no-retraining zip) | **0.375** — the true car was ranked 6th of 8 |
| **Cross-validated** | rank 1 in all five training cases that share the test case's signal schema (leave-one-case-out) |
| **Model** | pairwise faulty-vs-normal logistic ranker on per-signal, case-centred descriptors |
| **Evolution** | none for the final model: it is the *seed* program (the raw-signal programs ran 600 iterations) |

**This is the weakest of the four results, and the cross-validation was misleading.** Read "Why it failed" below.

## Data

Each case is one `.xlsx`: 8 cars sampled every 30 s (3,263–22,262 rows). Six training cases give the faulty car; one held-out case.
Cases expose **different signal sets**: five training cases and the held-out case share 8 signals per car (setting/running mode,
control temperature for cooling and heating, indoor and outdoor average temperature, load-halved, information-valid); training
case 4 records 63 unrelated signals per car. The held-out case is car model A / train 620 — the same model and train as training cases
1 and 2.

## Preprocessing (`acv_case_features`, `load_data`)

1. For every car and every signal it has: mean, std, missing-fraction and last-minus-first change of the valid readings.
2. **Centre on the case:** subtract, for each feature, the median over the 8 cars *of the same case*. Only "which car deviates
   from its siblings" remains, so case-wide offsets (weather, route, train) drop out.
3. **Union the columns** over all cases (148); a signal a case does not have is `NaN`.
4. A car is a 148-vector; label = 1 for the faulty car of its case, else 0. 48 labelled cars, all used.

## Model

At prediction time, only signals that the scored case actually exposes *and* that training observed are kept. Training turns every
faulty car into pairs `faulty − normal` (both signs, labels 1/0) against every healthy car, then fits
`StandardScaler → SelectKBest(f_classif, k=5) → LogisticRegression(C=0.003, class_weight="balanced")` on the differences. The
decision function of the fitted model is each car's fault score, and cars are ranked within their case.

Pairwise comparison suits the question ("which car ranks first?") and, in the final fit, turns the 6 faulty cars into 6 × 42 = 252 comparisons (× 2 signs) instead of 6 positive examples.

## Why we chose it, and why it failed

* The earlier evolved raw-signal programs collapse all signals of a car into per-timestep statistics, so they cannot see *which*
  signal deviates. They scored 0.95 under leave-one-case-out on the five same-schema cases; this ranker scored **1.000** and
  was adopted on that evidence.
* **On the one truly unseen case it ranked the faulty car 6th.** The leave-one-case-out result was optimistic: the design
  (per-signal features, `k=5`, `C`) had been developed by looking at these same six cases, and with five faulty cars,
  `SelectKBest` can pick up signals that do not generalise. It selected heating-temperature spread and outdoor-temperature spread, which
  single out car 04 — a car the public score shows was not the faulty one.
* A post-hoc look (using the faulty car implied by the public score, car 01) found the signature that *was* consistent: the
  faulty car's **cooling control temperature sits several robust standard deviations below its siblings** (case 2: −7.7,
  the held-out car 01: −6.2, siblings ≤ 1.3). The supervised ranker never selected that feature.
* Had the previous raw-signal ranking been kept it would have put car 01 second (0.875). That ranking was replaced
  because of the cross-validation evidence — a reminder that with six cases, a one-case improvement is noise-level evidence.
* **Ensembling** was tested and not used (nested out-of-fold 1.000 vs 1.000, a tie; schemes 0.96–1.00).

## What would improve it (not done)

A model with no or very few fitted parameters that scores "how far is this car from its siblings", using robust deviations of the
control-temperature signals, would not depend on five faulty examples. Adding the now-known faulty car of the public case as a seventh
training case is only legitimate if the final scoring uses a different set.

## Hard-label retraining (the "with" zip)

The held-out case is pseudo-labelled (top-ranked car = faulty) only if all 10 subsampled copies agree on its top car. Tuned by
leave-one-case-out on the labelled data: every cutoff scores 1.000, so nothing was gained and the strictest cutoff (all copies must
agree) is used. On the real case the top car stays 04 and two mid-ranked cars swap.

## Reproduce

```bash
python acv_model.py --data-root <02_Datasets> --out out/
```
Both output files are byte-identical to the ones in the submitted zips. (Separately, a *personal-check* zip in `submissions/` overwrites
this ranking with the answer implied by the public score; it is labelled as such and is not a model result.)
