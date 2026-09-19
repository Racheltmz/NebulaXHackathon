# Door — abnormal-resistance cycle classifier

`door_model.py` is the single file behind the Door submission: raw-data preprocessing, the exact OpenEvolve program that was
fitted, and a driver that trains on **all** labelled data and writes `door_predictions.csv`.

| | |
|---|---|
| **Task** | find every door open/close cycle in a continuous stream, label it `Normal` or `Abnormal resistance` |
| **Official metric** | IoU-weighted F1 of predicted vs true segments (timing and label) |
| **Public-test score** | **0.9474** (equal to 2 of the 38 segments wrong, if every segment boundary matches the reference) |
| **Cross-validated** | 1.000 (3-fold, 5-fold, train/test, and all-data 5-fold); the majority-class baseline scores 0.727 |
| **Model** | scaled RBF-SVM on 679 hand-built per-segment statistics |
| **Evolution** | 610 OpenEvolve iterations (on the mis-segmented data, see below) + 1 on the corrected data |

## Data

`Train.csv` is one continuous stream: 18,036 rows × 17 columns — a timestamp and 16 channels (motor current, motor voltage,
back-EMF, door-opening/closing times, open/close commands, four limit/lock switches, opened/locked/opening/closing flags, door-leaf
position). `Train_Segments_Answer.csv` gives the 110 true cycles (80 Normal, 30 Abnormal resistance; 137–190 rows each).
`Test.csv` (6,253 rows) is the same kind of stream without labels; the task is to find the cycles *and* classify them.

## Preprocessing (`load_data`, `segment_stream`)

1. **Channels** → numeric, missing → 0, `float32`. A segment is a `[16, T]` array.
2. **Training segments** are sliced by the *exact row indices* of each answer row's `start_time` / `end_time`
   (asserted: the row count equals the answer file's `n_rows` for all 110).
3. **Test segmentation.** Cycles are separated by idle gaps, so the stream is cut wherever the time between consecutive rows exceeds
   1 second. Timestamps are `Y-M-D-h-m-s-ms` with an unpadded millisecond field (`…-0-20` and `…-3-700`), so the millisecond
   part is parsed as an **integer**. This rule reproduces all 110 training segments exactly (start and end row) and finds 38 cycles
   in the test stream, covering all 6,253 rows with no overlap.
4. **Row order.** The 110 rows are used in a fixed seed-7 order (`LABEL_ORDER`). The model picks its hyper-parameters with an
   internal cross-validation whose folds depend on row order, so the order is pinned to reproduce the submitted predictions.

> **A bug found and fixed along the way.** The first Door arrays were built with a custom timestamp-to-integer conversion that
> mis-ordered millisecond fields of different widths: *none* of the 110 segments had the right length (a 186-row cycle came out
> as 1,979 rows). Every early Door number, classical and deep, was measured on garbled slices. After the fix the same program
> scored 1.000, so it had learned something real from the garbled data, but the earlier numbers were not trustworthy.

## Model

`StandardScaler → SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced")` on a 679-number hand-built feature vector per segment:

* **Per channel (35 numbers):** mean, std, min, max, range; 10/25/50/75/90th percentiles; mean |first difference|, mean square,
  linear slope, mean absolute deviation, first and last value, std of the difference, five percentiles of the difference, lag-1
  autocorrelation, four sub-segment means, and an 8-point resampled shape.
* **Shape profile:** every channel z-normalised and resampled to 16 points, then the mean and std across channels at each point.
* **Cross-channel profiles:** at each time step the mean, std, range, median, quartiles and difference-statistics across the 16
  channels, each resampled to 8 points, plus the step-to-step change and the low-frequency spectrum of the cross-channel mean.

## Why it works, and why it is the final choice

* An abnormal cycle (door jamming, deformed leaf, sticking strip) shows up as a different *shape and level* of motor current /
  voltage / position over the cycle, not as a single-row spike. Summaries over the whole segment — level, spread, slope,
  sub-segment means, a resampled shape — capture that, and an RBF-SVM separates the two clusters with very little data (110 rows).
* `class_weight="balanced"` handles the 80/30 imbalance.
* **Alternatives checked, in order of evidence:**
  * *Ensembling / fold schemes.* Six fold-creation schemes (all data, "all abnormal + a different third of the normals", overlapping
    2/3, cross-fit k-fold, bootstrap, 80% subsamples) were scored out of fold. The best any reached was 1.000, tying the single
    program; the "different third of the normals" scheme was the only one below (0.9955) because each member sees less data. The rule
    was that an ensemble is used only if its nested out-of-fold score is *strictly higher* than the single program trained on all
    the data, so the single program is used.
  * *Deep learning.* A frozen-TimesFM3 + gated-attention head on a GPU cluster reached 0.994 combined after two iterations on the
    corrected data. That does not beat 1.000, so the classical model is used.
  * *Earlier work* on the same task reported ~0.98 balanced accuracy, but its segment boundaries did not match the labelled ones (only 1 of 110 start times agreed).

## What we do not know

The 1.000 cross-validation score is at odds with the public score of 0.947 (two of 38 cycles wrong). Both training and test are single
continuous streams, so neighbouring cycles share operating conditions and the cross-validation is optimistic; the two errors have
not been diagnosed. There are only 30 abnormal training cycles.

## Reproduce

```bash
python door_model.py --data-root <02_Datasets> --out out/     # writes out/door_predictions.csv
```
The output is byte-identical to the submitted Door predictions.
