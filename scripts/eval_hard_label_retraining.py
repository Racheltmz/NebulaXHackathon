#!/usr/bin/env python3
"""Does hard-label retraining help the SINGLE fully-trained programs the submission uses?

Each outer fold plays the unlabelled private test set:
  before : program trained on the outer-train labelled data, predicts the fold
  after  : the fold is hard-labelled with those predictions (class / top-1 faulty car / predicted value),
           the program is retrained on outer-train + pseudo-labelled fold, and predicts the fold again
Both are scored against the hidden truth with the official metric, pooled over folds and repeated over
several outer-CV seeds.  usage: python scripts/eval_hard_label_retraining.py [--tasks ...] [--seeds 0 1 2]
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
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--reports", default="submission")
    a = ap.parse_args()
    out = {}
    for name in a.tasks:
        data = fd.LOADERS[name]()
        task = task_module(name).build(data, 10)          # all archive programs available by name
        prog = json.loads((REPO / a.reports / f"report_{name}.json").read_text())["program"]
        t = next(t for t in task.types if t.name == prog)
        y, groups = np.asarray(data.y_lab), data.groups_lab
        dt = object if task.kind == "class" else float
        rows = []
        for seed in (a.seeds if task.kind != "rank" else [0]):
            before, after = np.empty(len(y), dtype=dt), np.empty(len(y), dtype=dt)
            for tr, va in E.outer_splits(task, y, groups, 5, seed):
                Xtr, Xva = [data.X_lab[t.input_key][i] for i in tr], [data.X_lab[t.input_key][i] for i in va]
                gva = None if groups is None else groups[va]
                o1, err = E._run_member(t.code, Xtr, y[tr], Xva)
                assert err is None, err
                rep = E._rep_from_members(task, [o1], gva)
                p1 = E.ensemble_output(task, {"k": rep}, {"k": 1.0})
                yp = E.pseudo_labels(task, p1, gva)
                yall = np.concatenate([y[tr].astype(object), yp]) if task.kind == "class" else np.concatenate([y[tr], yp])
                o2, err = E._run_member(t.code, Xtr + Xva, yall, Xva)
                assert err is None, err
                p2 = E.ensemble_output(task, {"k": E._rep_from_members(task, [o2], gva)}, {"k": 1.0})
                before[va], after[va] = p1, p2
            rows.append((float(task.metric(y, before, groups)), float(task.metric(y, after, groups)),
                         float(np.mean(np.asarray(before) != np.asarray(after))) if task.kind == "class" else None))
        b, af = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
        out[name] = {"program": prog, "before": b, "after": af, "delta": af - b, "per_seed": rows}
        print(f"[{name}] {prog}: before retraining {b:.4f} -> after {af:.4f}  (delta {af - b:+.4f}); per seed {[(round(r[0], 4), round(r[1], 4)) for r in rows]}", flush=True)
    (REPO / a.reports / "hard_label_retraining_effect.json").write_text(json.dumps(out, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
