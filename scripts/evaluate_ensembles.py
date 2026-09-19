#!/usr/bin/env python3
"""Measure the ensemble pipeline honestly on the labelled data.

Each outer fold plays the role of the unlabelled private test set: the pipeline (diverse-fold
members -> bagged prediction -> hard-label retraining) is run on the remaining labelled data and scored
against the hidden true labels.  Reports, per task, the pooled official metric for
  single   the best single evolved program trained on all outer-train data (no ensembling)
  stage0   one diverse-fold partition, calibrated weights
  stage1   bagged diverse-fold ensemble (test-time bagging)
  stage2   stage1 + hard-label retraining on the fold   <- what the final CSVs use
Calibrated weights are read from submission/report_<task>.json (fitted on the same labelled data, so all
four numbers are slightly optimistic in the same way; their DIFFERENCES are the informative part).
usage: python scripts/evaluate_ensembles.py [--tasks door acv rail shm] [--bags 3]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E  # noqa: E402


def load_task_module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_ensemble", REPO / name / "ensemble.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def simulate(task, data, weights, n_outer, seed, bags):
    y = np.asarray(data.y_lab)
    groups = data.groups_lab
    flag = task.make_flag(y)(y)
    splits = (list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups)) if task.kind == "rank"
              else list(StratifiedKFold(n_outer, shuffle=True, random_state=seed + 7).split(np.zeros(len(y)), flag)))
    dt = object if task.kind == "class" else float
    pooled = {k: np.empty(len(y), dtype=dt) for k in ("stage0", "stage1", "stage2")}
    single = {t.name: np.empty(len(y), dtype=dt) for t in task.types}
    for s, (tr, va) in enumerate(splits):
        E.log(f"  [{task.name}] simulated private-test fold {s + 1}/{len(splits)}")
        Xtr = {k: [v[i] for i in tr] for k, v in data.X_lab.items()}
        Xva = {k: [v[i] for i in va] for k, v in data.X_lab.items()}
        ytr, gva = y[tr], (None if groups is None else groups[va])
        for t in task.types:
            out, err = E._run_member(t.code, Xtr[t.input_key], ytr, Xva[t.input_key])
            single[t.name][va] = out if err is None else (np.nan if dt is float else "Normal")
        r0, _ = E.train_predict(task, Xtr, ytr, Xva, gva, seed + s)
        pooled["stage0"][va] = E.ensemble_output(task, r0, weights)
        r1, _ = E.train_predict_bagged(task, Xtr, ytr, Xva, gva, seed + 100 + s, bags)
        o1 = E.ensemble_output(task, r1, weights)
        pooled["stage1"][va] = o1
        yp = E.pseudo_labels(task, o1, gva)
        yall = np.concatenate([ytr.astype(object), yp]) if task.kind == "class" else np.concatenate([ytr, yp])
        Xall = {k: list(Xtr[k]) + list(Xva[k]) for k in Xtr}
        r2, _ = E.train_predict_bagged(task, Xall, yall, Xva, gva, seed + 200 + s, bags)
        pooled["stage2"][va] = E.ensemble_output(task, r2, weights)
    m = lambda o: float(task.metric(y, o, groups))
    res = {k: m(v) for k, v in pooled.items()}
    res["single_by_program"] = {k: m(v) for k, v in single.items() if not (dt is float and np.isnan(v).any())}
    res["best_single"] = max(res["single_by_program"].values())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["door", "acv", "rail", "shm"])
    ap.add_argument("--bags", type=int, default=3)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--out", default="submission")
    a = ap.parse_args()
    results = {}
    for name in a.tasks:
        mod = load_task_module(name)
        data = {"door": None}  # placeholder to keep flake quiet
        from common import final_data as fd
        data = fd.LOADERS[name]()
        task = mod.build(data, 3)
        rp = REPO / a.out / f"report_{name}.json"
        rep = json.loads(rp.read_text()) if rp.exists() else None
        weights = rep["weights"] if rep else {t.name + "|div": 1.0 for t in task.types}
        if rep:
            task.flag_scale, task.out_scale = rep.get("flag_scale", 1.0), rep.get("out_scale", 1.0)
        results[name] = simulate(task, data, weights, a.outer, 0, a.bags)
        r = results[name]
        print(f"\n[{name}] single best program {r['best_single']:.4f} | one partition {r['stage0']:.4f} | "
              f"bagged ensemble {r['stage1']:.4f} | + hard-label retrain {r['stage2']:.4f}", flush=True)
    (REPO / a.out).mkdir(parents=True, exist_ok=True)
    (REPO / a.out / "simulation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
