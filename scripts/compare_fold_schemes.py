#!/usr/bin/env python3
"""Compare fold-creation schemes on identical out-of-fold splits.

For each task: prune weak program types, then run the outer CV once per seed with EVERY scheme
(full, div3, ovl, kfold, boot, sub80) and report the pooled official metric for
  * each (type, scheme) key,
  * each scheme with the surviving types averaged with uniform weights,
  * all schemes together (uniform),
averaged over the outer-CV seeds.  `full` on the best type is the "best single program" baseline.
usage: python scripts/compare_fold_schemes.py --tasks door acv rail shm [--seeds 0 1]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402


def task_module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_ens", REPO / name / "ensemble.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["door", "acv", "rail", "shm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--out", default="submission")
    a = ap.parse_args()
    results = {}
    for name in a.tasks:
        data = fd.LOADERS[name]()
        task = task_module(name).build(data, 3)
        y, groups = np.asarray(data.y_lab), data.groups_lab
        _, kept = E.prune_types(task, data.X_lab, y, groups, 5, 0, keep=3)
        E.log(f"[{name}] types kept: {kept}")
        per_seed = []
        for seed in a.seeds:
            reps = E.oof_representations(task, data.X_lab, y, groups, 5, seed, schemes=E.ALL_SCHEMES)
            def score(keys):
                keys = [k for k in keys if k in reps and not np.isnan(reps[k]).any()]
                w = {k: 1.0 / len(keys) for k in keys}
                return float(task.metric(y, E.ensemble_output(task, {k: reps[k] for k in keys}, w), groups))
            row = {"keys": {k: score([k]) for k in reps}}
            for s in E.ALL_SCHEMES:
                row[f"scheme:{s}"] = score([f"{t.name}|{s}" for t in task.types])
            row["all_schemes_uniform"] = score(list(reps))
            row["best_single_program_full"] = max(score([f"{t.name}|full"]) for t in task.types)
            per_seed.append(row)
        agg = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0] if k != "keys"}
        agg["keys"] = {k: float(np.mean([r["keys"][k] for r in per_seed])) for k in per_seed[0]["keys"]}
        results[name] = agg
        print(f"\n[{name}] mean over outer seeds {a.seeds}")
        print(f"   best single program (all data): {agg['best_single_program_full']:.4f}")
        for s in E.ALL_SCHEMES:
            print(f"   scheme {s:6s} (types averaged): {agg['scheme:' + s]:.4f}")
        print(f"   all schemes uniform            : {agg['all_schemes_uniform']:.4f}", flush=True)
    out = REPO / a.out
    out.mkdir(exist_ok=True)
    (out / "fold_scheme_comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
