# Corrected multivariate TimesFM3 head sweep

## Hypothesis

Using TimesFM3 in variate-attention mode on the pooled UCR+UEA corpus, a shared attention or temporal head with dataset-specific channel and label-query tokens will improve held-out whole-dataset classification over the existing linear-head baseline and provide a stronger initialization for PS3 fine-tuning.

## Contract

- Data: all available examples from 128 UCR plus 21 completed UEA datasets; 27 FlaMinGo datasets are evaluation-only and 9 incomplete UEA datasets are deferred.
- Backbone: frozen TimesFM3, corrected `[B,C,N,P]` multivariate extraction with all observed channels marked target.
- Primary metrics: held-out balanced accuracy, macro-F1, accuracy; report mean across OOD datasets.
- Comparators: existing FlaMinGo results where available and prior 41-dataset TimesFM head runs.
- Variants: linear, temporal, pool, SDA attention; qpc=4; balanced CE; feature normalization.
- High-channel safeguard: SDA attention pools only the channel axis after channel-specific offsets to at most 128 memory channel groups; low-channel datasets are unchanged.
- Cache safeguard: TimesFM3 still processes every original channel, but datasets above eight channels are stored as four post-backbone channel-distribution tokens (mean/std/max/min) to avoid >100 GB per-dataset caches.
- Budget: four parallel GPU jobs, each under two hours.
- Stop gate: retain only runs with durable metrics and complete logs; no claim of success until an untouched OOD evaluation and PS3 evaluation verify it.

## Execution

1. Use completed UCR/UEA feature pairs directly; stream only the nine deferred UEA datasets later.
2. Train the shared head on 100% of eligible source examples, with no source 90/10 split.
3. Run four head variants in parallel on a held-out OOD dataset, then repeat the best on independent OOD datasets.
4. Fine-tune the selected head on stratified PS3 folds and evaluate Door/Rail/ACV/SHM under official metrics.

## Revision log

- 2026-09-18: switched from old univariate-style cache to corrected TimesFM3 variate-attention extraction and added balanced accuracy to evaluation.
