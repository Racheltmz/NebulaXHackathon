# TimesFM3 Foundational Classifier — Progress Report

Updated: 2026-09-18

## Executive status

The project is active and has not reached the success criteria yet.

All model/data execution has been remote through the NUS SoC VPN and SLURM project:

- Remote project: `/home/s/saad304/timesfm-classifier`
- Backbone: frozen TimesFM3 feature caches
- Main cached representations: short-context `data/features` and long-context `data/features1024`
- Experiment policy: four parallel jobs per round, each capped below two hours

The strongest general-purpose direction so far is a frozen TimesFM feature extractor with a shared linear classifier head followed by task-local adaptation. The long-context cache helps Ham and CBF, but the four-dataset mean remains below FlaMinGo.

## OOD classification comparison

FlaMinGo model-card reference values used for comparison are approximately: ECG200 0.900, Ham 0.695, FordA 0.931, Wafer 0.995, CBF 0.994, and ElectricDevices 0.700. The six-task mean is approximately 0.869.

| Model/configuration | ECG200 | Ham | CBF | ElectricDevices | Mean on four holdouts |
|---|---:|---:|---:|---:|---:|
| FlaMinGo reference | 0.900 | 0.695 | 0.994 | 0.700 | 0.822 |
| TimesFM short-context linear adaptation | 0.920 | 0.629 | 0.941 | 0.621 | 0.778 |
| TimesFM long-context linear adaptation | 0.900 | 0.724 | 0.947 | 0.620 | 0.798 |
| TimesFM long-context, 20-epoch repurposing + adaptation | 0.920 | 0.705 | 0.970 | 0.634 | 0.807 |
| TimesFM short+long multiscale adaptation | 0.910 | 0.648 | 0.939 | 0.634 | 0.783 |
| TimesFM long-context + train-only normalization | 0.930 | 0.705 | 0.931 | 0.526 | 0.773 |
| Temporal-statistics global head | 0.760 | 0.571 | 0.809 | 0.518 | 0.665 |
| Prototype metric head | 0.690 | 0.486 | 0.333 | 0.475 | 0.496 |
| SDA attention, 20 repurposing epochs | 0.860 | 0.667 | 0.802 | 0.346 | 0.669 |
| SDA with prototype query initialization | 0.460 | 0.505 | 0.341 | 0.146 | 0.363 |

The latest 20-epoch long-context adaptation result is the current best TimesFM configuration. It reaches 0.970 on CBF and improves the four-holdout mean to approximately 0.807, but it still does not beat FlaMinGo on the comparable average. ElectricDevices remains the main OOD weakness.

The latest class-imbalance exponent sweep did not improve that result: ECG200 0.930, Ham 0.724, CBF 0.961, and ElectricDevices 0.598, for a mean of approximately 0.803.

## Data volume actually used

The main generalized head-training cohort was the 41-dataset UCR collection:

- 23,352 training examples across 41 datasets
- 51,319 held-out UCR test examples
- 74,671 examples total
- approximately 3.1 GB of short-context cached TimesFM features
- approximately 6.1 GB of long-context cached TimesFM features

Each OOD experiment held out one entire UCR dataset. Therefore the shared head saw roughly 22–23 thousand source training examples per run, but never the held-out dataset's training or test examples during shared repurposing. The separate PS3 data contains 391 examples across Door, Rail, ACV, and SHM, and was used only for downstream fine-tuning/evaluation rather than the generalized UCR repurposing cohort.

## PS3 fixed held-out results

These are the fixed train/test split results currently available. They should not be treated as robust generalization evidence without the cross-validation results below.

| Task | Current best fixed-split result | Metric | Notes |
|---|---:|---|---|
| Door | 0.9688 | Balanced accuracy | Classical compact/raw feature SVC; historical verified run |
| Rail | 0.9262 | Balanced accuracy | Compact engineered SVC `svc05g2`; 217 train / 55 test |
| ACV | 1.0000 | Fault rank / rank-1 | Classical run had one held-out case; weak evidence |
| SHM | 0.6061 | Official score = 1 - MAPE | Classical ExtraTrees; 51 train / 13 test |

Current TimesFM-only PS3 fine-tuning is weaker:

| Task | TimesFM-head result |
|---|---:|
| Door | 0.9375 BAcc |
| Rail | 0.6000 BAcc |
| ACV | 0.5000 BAcc |
| SHM | 0.4353 score |

The recent TimesFM+raw Rail fusion experiments were also negative, ranging from 0.533 to 0.704 BAcc. Rail augmentation variants peaked at 0.9191, below the unaugmented 0.9262 baseline. The SVC ensemble/calibration sweep peaked at 0.9121.

## PS3 3-fold and 5-fold cross-validation audit

This audit was run remotely on the PS3 training split using compact engineered features and a balanced SVC for classification, and log-target ExtraTrees for SHM. It is a training-set cross-validation audit, not the fixed test-set score.

| Task | 3-fold result | 5-fold result | Interpretation |
|---|---:|---:|---|
| Door | BAcc 0.8464; macro-F1 0.8283 | BAcc 0.8496; macro-F1 0.8496 | Much lower than fixed-test 0.9688 |
| Rail | BAcc 0.6212; macro-F1 0.5881 | BAcc 0.6800; macro-F1 0.6457 | Indicates high split sensitivity and minority-class instability |
| ACV | BAcc 0.4714; macro-F1 0.4521 | BAcc 0.4714; macro-F1 0.4521 | Fixed-test 1.0 is not reliable evidence of broad generalization |
| SHM | MAPE 0.2988; score 0.7012 | MAPE 0.2898; score 0.7102 | Better than the fixed-test classical score, but small-sample uncertainty is high |

Class counts in the training split were Door 60/23, Rail 187/11/19, and ACV 35/5. In particular, Rail and ACV have too few minority examples for stable fold estimates; these results are nevertheless important because they expose the gap between one fixed split and broader validation.

## Main experiment ledger

Completed directions and conclusions:

- Frozen TimesFM3 + ordinary linear head: strongest general baseline, but below FlaMinGo mean.
- Balanced versus unbalanced loss: balanced training was generally better; not sufficient.
- Query counts 1 versus 4: four queries generally better; not sufficient.
- Pool head and adaptation-head variants: CBF improved in some runs, but mean remained below target.
- Deeper task-local adaptation: strongest head strategy so far; improves ECG/CBF and long-context Ham.
- Prototype initialization for linear head: harmful.
- Prototype metric-learning head: harmful across all four holdouts.
- Temporal-statistics global head: below the linear baseline.
- SDA/shared attention head: below linear adaptation, including after 20 epochs.
- SDA raw prototype query initialization: severely harmful.
- Short+long feature fusion: modest ElectricDevices improvement, but lower mean than long-only.
- Train-only feature normalization: helps ECG/Ham but harms ElectricDevices.
- Raw-only and TimesFM+raw fusion: rejected for OOD transfer and Rail.
- Rail SVC C/gamma sweeps, calibrated ensembles, stacking, PCA, XGBoost, ROCKET, KNN, and MUSE: none beat the compact SVC baseline.
- Rail minority augmentation: rejected.
- SHM regressors: latest scores were ExtraTrees 0.5874, ExtraTrees(min-leaf 2) 0.5794, HistGradientBoosting 0.5662, and RandomForest 0.5438; the previous classical ExtraTrees baseline remains 0.6061.

## Current interpretation

1. The frozen TimesFM representation is useful for ECG/CBF and partially useful for Ham, but it is not yet a universally transferable classification representation.
2. Dataset-local adaptation is materially more useful than a global classifier head.
3. The PS3 fixed test split is optimistic relative to 3/5-fold training validation, especially for Rail and ACV.
4. The main unresolved problems are ElectricDevices transfer, Rail minority-class robustness, ACV ranking robustness, and SHM regression.

## References

- FORMED / shared attention repurposing: https://arxiv.org/html/2410.03794v2
- FlaMinGo-timesfm model card: https://huggingface.co/PartAI/FlaMinGo-timesfm
- NebulaX PS3 specification: https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement

## Success status

Not achieved. The project has not yet demonstrated both:

- an average OOD classification score above FlaMinGo; and
- verified 0.95–0.99 BAcc across the requested PS3 classification tasks with robust validation.
