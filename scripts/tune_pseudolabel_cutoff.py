#!/usr/bin/env python3
"""Find the best confidence cutoff for hard-label retraining, using only the LABELLED data.

Each outer fold plays the unlabelled test set.  For the chosen single program:
  stage 1     the program (trained on the outer-train data) predicts the fold
  confidence  10 subsampled copies (80% of the outer-train rows) predict the fold; an item's confidence is
                class : fraction of copies that agree with the stage-1 label      (0.9 == the "p>=0.9 or <=0.1" band)
                rank  : fraction of copies whose top-1 car equals the stage-1 top-1 car (per case)
                reg   : consistency = 1 / spread of the copies' log-predictions; the cutoff is the FRACTION of
                        most-consistent traces that get pseudo-labels
  retrain     only items at/above the cutoff are pseudo-labelled (stage-1 labels/values) and added; the program
              is retrained on outer-train + those items and predicts the WHOLE fold again
Scored against the hidden truth with the official metric, pooled over the 3 outer folds (ACV: leave-one-case-out),
next to "no retraining".  The reported cutoff is the best-scoring one (ties -> the stricter cutoff).
usage: python scripts/tune_pseudolabel_cutoff.py --tasks door acv rail shm [--report rail=submission_refresh2 ...]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402

GRID = {"class": [0.0, 0.6, 0.7, 0.8, 0.9, 1.0], "rank": [0.0, 0.6, 0.8, 1.0], "reg": [1.0, 0.75, 0.5, 0.25]}
K_MEMBERS = 10


def task_module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_ens", REPO / name / "ensemble.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def copies(task, code, Xtr, ytr, Xp, groups_tr, seed, key, n_jobs):
    """K subsampled copies (80% of the training rows) -> their outputs on Xp."""
    rng = np.random.default_rng(seed)
    flag = task.make_flag(ytr)(ytr)
    subsets = []
    while len(subsets) < K_MEMBERS:
        subsets += E.scheme_subsets("sub80", flag, groups_tr, rng)
    subsets = subsets[:K_MEMBERS]
    res = Parallel(n_jobs=n_jobs)(delayed(E._run_member)(code, [Xtr[i] for i in s], ytr[s], Xp) for s in subsets)
    return [o for o, e in res if e is None]


def confidence(task, stage1, outs, groups_p):
    n = len(stage1)
    if task.kind == "class":
        return np.mean([np.asarray(o).astype(str) == np.asarray(stage1).astype(str) for o in outs], axis=0)
    if task.kind == "rank":
        conf = np.zeros(n)
        for g in np.unique(groups_p):
            idx = np.flatnonzero(groups_p == g)
            top = idx[np.argmax(stage1[idx])]
            conf[idx] = np.mean([idx[np.argmax(np.asarray(o, dtype=float)[idx])] == top for o in outs])
        return conf
    L = np.log(np.maximum(np.asarray(outs, dtype=float), 1e-6))
    return -L.std(axis=0)                     # higher = more consistent


def select(task, conf, cutoff):
    """Indices of items that receive a pseudo-label at this cutoff."""
    if task.kind == "reg":
        k = max(1, int(round(len(conf) * cutoff)))
        return np.argsort(-conf)[:k]
    return np.flatnonzero(conf >= cutoff - 1e-12)


def retrain(task, code, Xtr, ytr, Xp, stage1, chosen, groups_p, groups_tr):
    if len(chosen) == 0:
        return stage1
    if task.kind == "rank":            # a case is pseudo-labelled as a whole: top-1 car = 1, the others = 0
        cases = np.unique(groups_p[chosen])
        rows = np.flatnonzero(np.isin(groups_p, cases))
        yp = E.pseudo_labels(task, stage1, groups_p)[rows]
    else:
        rows = chosen
        yp = E.pseudo_labels(task, stage1, groups_p)[rows]
    yall = np.concatenate([ytr.astype(object), yp]) if task.kind == "class" else np.concatenate([ytr, yp])
    o, err = E._run_member(code, list(Xtr) + [Xp[i] for i in rows], yall, Xp)
    assert err is None, err
    return E.ensemble_output(task, {"k": E._rep_from_members(task, [o], groups_p)}, {"k": 1.0})


def tune(name, report_dir, seeds, n_jobs, folds):
    data = fd.LOADERS[name]()
    task = task_module(name).build(data, 10)
    prog = json.loads((REPO / report_dir / f"report_{name}.json").read_text())["program"]
    t = next(x for x in task.types if x.name == prog)
    y, G = np.asarray(data.y_lab), data.groups_lab
    grid = GRID[task.kind]
    dt = object if task.kind == "class" else float
    per_seed = []
    for seed in (seeds if task.kind != "rank" else [0]):
        base = np.empty(len(y), dtype=dt)
        pooled = {c: np.empty(len(y), dtype=dt) for c in grid}
        n_pl = {c: 0 for c in grid}
        for s, (tr, va) in enumerate(E.outer_splits(task, y, G, folds, seed)):
            Xtr, Xva = [data.X_lab[t.input_key][i] for i in tr], [data.X_lab[t.input_key][i] for i in va]
            gva, gtr = (None if G is None else G[va]), (None if G is None else G[tr])
            o, err = E._run_member(t.code, Xtr, y[tr], Xva)
            assert err is None, err
            stage1 = E.ensemble_output(task, {"k": E._rep_from_members(task, [o], gva)}, {"k": 1.0})
            base[va] = stage1
            conf = confidence(task, stage1, copies(task, t.code, Xtr, y[tr], Xva, gtr, seed * 100 + s, t.name, n_jobs), gva)
            for c in grid:
                chosen = select(task, conf, c)
                n_pl[c] += len(chosen)
                pooled[c][va] = retrain(task, t.code, Xtr, y[tr], Xva, stage1, chosen, gva, gtr)
            E.log(f"  [{name}] seed {seed} fold {s + 1} done")
        per_seed.append({"none": float(task.metric(y, base, G)),
                         **{str(c): float(task.metric(y, pooled[c], G)) for c in grid},
                         "n_pseudolabelled": {str(c): n_pl[c] for c in grid}})
    mean = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0] if k != "n_pseudolabelled"}
    best = max(mean[str(c)] for c in grid)
    tied = [c for c in grid if mean[str(c)] >= best - 1e-9]
    chosen_cut = (max(tied) if task.kind != "reg" else min(tied))       # stricter cutoff on ties
    return {"program": prog, "kind": task.kind, "grid": grid, "mean_metric": mean, "best_cutoff": chosen_cut,
            "best_metric": best, "no_retraining": mean["none"], "beats_no_retraining": bool(best > mean["none"] + 1e-9),
            "n_pseudolabelled_per_cutoff": per_seed[0]["n_pseudolabelled"], "per_seed": per_seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["door", "acv", "rail", "shm"])
    ap.add_argument("--report", nargs="*", default=[], help="task=dir holding report_<task>.json (default: submission)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", type=int, default=3, help="outer CV folds (ACV always leave-one-case-out)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="submissions")
    a = ap.parse_args()
    dirs = dict(x.split("=") for x in a.report)
    out = REPO / a.out
    out.mkdir(exist_ok=True)
    path = out / "pseudolabel_cutoff_tuning.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in a.tasks:
        r = tune(name, dirs.get(name, "submission"), a.seeds, a.jobs, a.folds)
        results[name] = r
        print(f"\n[{name}] {r['program']}  (cutoff meaning: {'fraction of copies agreeing' if r['kind'] != 'reg' else 'fraction of most-consistent traces kept'})")
        print(f"   no retraining : {r['no_retraining']:.4f}")
        for c in r["grid"]:
            print(f"   cutoff {c:<5}: {r['mean_metric'][str(c)]:.4f}   (pseudo-labelled items over 1 seed: {r['n_pseudolabelled_per_cutoff'][str(c)]})")
        print(f"   BEST cutoff {r['best_cutoff']}  metric {r['best_metric']:.4f}  beats no-retraining: {r['beats_no_retraining']}", flush=True)
        path.write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
