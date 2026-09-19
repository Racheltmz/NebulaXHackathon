# ACV — development code and model

Model code (`acv_classical.py`) is vendored from the `ian-model` branch's OpenEvolve search
(classical track) — an LLM-evolved `Model.fit/predict` program, not hand-written from scratch.
See `app/backend/ml/models/acv_classical.py` in the main app for the canonical copy and
`app/backend/ml/acv.py` for how it's called.

- **`code/acv_classical.py`** — the model (`class Model`: `fit`/`predict`).
- **`code/featurize.py`** — raw-file -> the array shape this model expects.
- **`code/fit_models.py`** — fits all four subsystems' models against the bundled PS3 training
  data and writes the artifact in `model/`. Run from the repo root: `python code/fit_models.py acv`.
- **`model/acv_classical.joblib`** — the fitted model this app currently runs in production
  (evolved classical (linear + retrieval + centroid)).

The model ships no trained weights of its own — every `ian-model` program retrains inside
`fit()` — so this joblib is the result of running `fit_models.py` once against
`PS3/02_Datasets/`, not something copied from that branch.
