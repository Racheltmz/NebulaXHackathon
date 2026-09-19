"""Evolved models vendored verbatim from the `ian-model` branch.

Each file is an OpenEvolve-discovered program exposing the frozen contract:

    class Model:
        def fit(self, X, y): ...      # X: list of [C, T] float arrays
        def predict(self, X): ...     # classes | values | per-item score

They are copied unchanged so their behaviour stays identical to the numbers reported on that
branch. Do not edit them here — re-export from `ian-model` instead.

One exception: that branch's sandbox pins numpy 1.26, and this app runs numpy 2.x, which removed
`ndarray.ptp()`. `door_classical.py` now calls `np.ptp(v)` instead. Same result, same values — it
is the only edit made to any vendored file. The raw-file -> [C, T]
transform each one expects lives next to it in `../featurize.py`, ported from that branch's
`data/ps3_prepare.py`.

There are no trained weights on that branch (every model retrains in `fit()`), so
`scripts/fit_models.py` fits them once against the bundled PS3 training data and writes the
artifacts this app loads.
"""
