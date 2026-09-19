# TimesFM3 classifier repurposing project

Remote root: `/home/s/saad304/timesfm-classifier`

The TimesFM3 backbone is frozen. The preparation job cached its transformer outputs for eight UCR datasets under `data/features/`. The shared probes and dataset-local channel/class tokens are trained by `src/train.py`.

## Completed remote experiments

All jobs used H100-47 GPUs and completed under two minutes after environment setup.

| Holdout | Variant | Accuracy | Macro-F1 | Checkpoint |
|---|---|---:|---:|---|
| ECG200 | linear | 0.660 | 0.558 | `reports/linear-ECG200.json` |
| ECG200 | pool | 0.650 | 0.498 | `reports/pool-ECG200.json` |
| ECG200 | sda | 0.630 | 0.470 | `reports/sda-ECG200.json` |
| ECG200 | sda_big | 0.620 | 0.539 | `reports/sda_big-ECG200.json` |
| Ham | pool | 0.629 | 0.620 | `checkpoints/pool-Ham.pt` |
| Ham | sda | 0.543 | 0.543 | `checkpoints/sda-Ham.pt` |
| Ham | sda_big | 0.438 | 0.435 | `checkpoints/sda_big-Ham.pt` |
| Ham | linear | 0.514 | 0.356 | `checkpoints/linear-Ham.pt` |

The strongest observed probe is the pooled metric head on Ham (0.620 macro-F1), while the compact shared decoder is the most structurally expressive choice for adapting to varying channel/class counts. The ECG200 and Ham evaluations are held-out cross-dataset adaptation tests: the shared head is trained on the other seven datasets and only new tokens are adapted on the held-out train split.

PartAI/FlaMinGo’s published 200M UCR reference includes ECG200 90.0%, Ham 69.5%, FordA 93.1%, Wafer 99.5%, CBF 99.4%, and ElectricDevices 70.0%. Those are dataset-specific published scores; exact weight reproduction requires access to its gated checkpoint.

## PS3 handoff

The official PS3 data is at `data/NebulaX-Hackathon-ProblemStatement/PS3`, with `data/ps3_manifest.txt`. Use `src/ps3_adapter.py` for per-subsystem parsing and `src/finetune_ps3.py` after producing a cached feature file with `features` and `labels` fields. Use `--mode classification` for Door/ACV/Rail and `--mode regression` for SHM. ACV ranking should sort per-car scores before writing `ranked_cars`; Door must first segment the continuous stream and write timestamped predictions.

The implementation follows the frozen-backbone/shared decoder strategy in [FORMED](https://arxiv.org/html/2410.03794v2). TimesFM3 weights are used for research/hackathon work only; verify the current non-commercial checkpoint license before production use.
