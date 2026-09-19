# SHM — cumulative fatigue-damage regression

`shm_model.py` is the single file behind the SHM submission: raw-data preprocessing (feature extraction from the stress traces), the
exact OpenEvolve program that was fitted, and a driver that trains on **all** 64 labelled traces and writes `shm_predictions.csv`.

| | |
|---|---|
| **Task** | predict one cumulative-damage number for each dynamic-stress trace |
| **Official metric** | `max(0, 1 − MAPE)` — *relative* error, so small damages matter as much as large ones |
| **Public-test score** | **0.9322** |
| **Cross-validated** | 0.924 (5-fold, pooled over all 64 traces); 0.940 combined score in the evolution protocol |
| **Model** | log-target kernel / SVR / ridge blend on 838 descriptors + a clipped residual calibrator |
| **Evolution** | 160 OpenEvolve iterations kept (158 attempted) on top of a seed program; the program used was found at iteration 41 |

## Data

64 training traces (damage 0.029–0.928) and 16 held-out traces. Each is a single column of 581,120 dynamic-stress samples from
a measurement point on a rail vehicle, from two lines and two load conditions (AW0 and AW4); all samples are healthy operation, and the
label is the damage accumulated by the trace (Miner's rule with rainflow counting). File numbers are arbitrary.

## Preprocessing (`shm_vector` and the feature functions)

Each trace becomes an 838-vector, **[ multiscale (561) | generic (44) | rainflow (83) | temporal (150) ]**, computed trace by trace:

* **Multiscale (561).** The trace is cut into 8, 16, 32, 64 and 128 equal windows; per window: mean, std, RMS, peak-to-peak, mean |x|,
  max |x|, and std of the sample-to-sample difference (a roughness proxy). Each of those 7 series is summarised by 8 quantiles, first,
  last, delta, std over time, max, min, slope and area under the curve.
* **Generic (44).** Global moments, quantiles, mean-crossing rate, turning-point cycle ranges from the 16×-downsampled trace, and spectral
  band fractions of the block-averaged signal.
* **Rainflow (83).** Four-point rainflow cycle counting on the trace's reversals at four down-sampling factors (4, 16, 64, 256): cycle
  count, range quantiles, mean/std of ranges, and **Miner-style damage sums** Σ(count × rangeᵐ) for exponents m = 1…12 — the physical
  quantity the target is built from.
* **Temporal (150).** 128 windows; for seven series their quantiles, early/late means and ratios, slopes over the whole trace and each
  half, and the area — how the load *evolves* over the record.
* **Row order** is pinned to a fixed seed-7 order (`LABEL_ORDER`) because the model's internal cross-validation folds depend on it.

## Model

All learners are fitted on `log(damage + offset)` with sample weights ∝ damage^−½, so the training loss tracks *relative* error:

* two multiscale RBF kernel-ridge models (top-20 and top-10 features by mutual information; blended 70/30);
* two kernel-ridge models on the generic features, one on the rainflow features (top-3), an SVR on the temporal features (top-10);
* a strongly regularised **linear ridge** view on the top-12 of generic + rainflow + temporal;
* blend: `0.88 × [0.4·multiscale + 0.6·(0.4·rel + 0.42·generic + 0.108·rainflow + 0.072·temporal)] + 0.12 × linear`.

**Residual calibrator.** Five-fold out-of-fold predictions of the blend give log residuals, which a ridge model (α = 0.02) regresses on
ten log amplitude / roughness covariates and the log prediction. At prediction time the correction is clipped to ±1.25 and applied at
60% strength: `pred = blend × exp(0.6 × correction)`.

## Why it works, and why it is the final choice

* **Fit what is scored.** MAPE is relative, so the log target and the damage^−½ weights stop the few high-damage traces from
  dominating.
* **Physics-shaped features.** Damage is a sum of powers of cycle ranges, so rainflow ranges and Miner sums, plus amplitude
  measured at several time scales, carry the signal; with 64 traces, only a few mutual-information-selected features per view are used.
* **Diversity in a tiny sample.** Kernels with different bandwidths and feature views, plus a linear view for when the kernels
  extrapolate badly, average out the variance of any one small fit.
* **The calibrator** removes systematic bias (low/high-damage asymmetry) using strictly out-of-fold residuals.
* **Alternatives (all-data 5-fold):**
  * programs on the raw traces: 0.82–0.83, against 0.924 for the descriptor programs;
  * a GPU deep track: 0.63 combined score;
  * **ensembling**: nested out-of-fold 0.9237 against 0.9247 for the single program, so the single program is used. Fold-creation
    schemes: 0.909 ("all high-damage + a different third of the rest") to 0.930 (cross-fit k-fold), all within noise of the single
    program (0.928) except the weakest;
  * evolution moved the seed from 0.930 to 0.940 combined; the final program was chosen on all-data 5-fold score, where several
    evolved programs tie at 0.924.

## What we do not know

The descriptors come from an earlier effort that designed them by looking at all 64 traces, so 0.924 is optimistic; the public score is 0.932.
With 64 traces, differences of a few thousandths between programs are noise.

## Reproduce

```bash
python shm_model.py --data-root <02_Datasets> --out out/ --jobs 4     # ~10 min: recomputes the features from the raw files
```
The output matches the submitted SHM predictions to about 5×10⁻⁸ relative (float rounding in the recomputed features).
