#!/usr/bin/env python3
"""Build two submissions from the decided best approach per task:
   submissions/no_hard_label_retraining/   models fitted on ALL labelled data, predict the held-out set
   submissions/with_hard_label_retraining/ the held-out set is hard-labelled with those predictions
                                           (class / top-1 faulty car / predicted value), the same models are
                                           retrained on labelled + pseudo-labelled data and predict again
Each folder gets the four organiser CSVs, predictions.zip, final_predictions.csv and manifest.json.
The approach per task comes from a report_<task>.json (mode 'single fully-trained program' or 'ensemble').
usage: python scripts/build_two_submissions.py door=submission acv=submission rail=submission_refresh2 shm=submission_refresh2
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import ensemble_lib as E, final_data as fd  # noqa: E402

VARIANTS = {"no_hard_label_retraining": "stage1", "with_hard_label_retraining": "final"}
CUTOFFS = REPO / "submissions" / "cutoff_3fold" / "pseudolabel_cutoff_tuning.json"


def tune_module():
    spec = importlib.util.spec_from_file_location("tune_cut", REPO / "scripts" / "tune_pseudolabel_cutoff.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def task_module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_ens", REPO / name / "ensemble.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def predict_both(name, report):
    mod = task_module(name)
    data = fd.LOADERS[name]()
    task = mod.build(data, 10)                                  # every archive program available by name
    y, groups, gt = np.asarray(data.y_lab), data.groups_lab, data.groups_test
    if report["mode"].startswith("single"):
        prog = report["program"]
        task.types = [t for t in task.types if t.name == prog]
        task.schemes, weights, bags = ["full"], {f"{prog}|full": 1.0}, 1
        task.flag_scale = task.out_scale = 1.0
    else:
        task.types = [t for t in task.types if t.name in report["program_types"]]
        task.schemes, weights, bags = report["schemes"], report["weights"], report["bags"]
        task.flag_scale, task.out_scale = report.get("flag_scale", 1.0), report.get("out_scale", 1.0)
    assert task.types, f"{name}: program(s) not found in the archive"
    reps1, err = E.train_predict_bagged(task, data.X_lab, y, data.X_test, gt, 100, bags, groups)
    assert not err, err
    out1 = E.ensemble_output(task, reps1, weights)
    info = {}
    if report["mode"].startswith("single"):
        # confidence-cutoff hard-label retraining, cutoff tuned on the labelled data (scripts/tune_pseudolabel_cutoff.py)
        T = tune_module()
        cut = json.loads(CUTOFFS.read_text())[name]["best_cutoff"]
        t = task.types[0]
        Xl, Xt = data.X_lab[t.input_key], data.X_test[t.input_key]
        copies = T.copies(task, t.code, Xl, y, Xt, groups, 7, t.name, 3)
        conf = T.confidence(task, out1, copies, gt)
        chosen = T.select(task, conf, cut)
        out2 = T.retrain(task, t.code, Xl, y, Xt, out1, chosen, gt, groups)
        info = {"cutoff": cut, "pseudo_labelled": int(len(chosen)), "heldout_items": int(len(out1)),
                "copies_used": len(copies)}
    else:
        yp = E.pseudo_labels(task, out1, gt)
        y_all = np.concatenate([y.astype(object), yp]) if task.kind == "class" else np.concatenate([y, yp])
        X_all = {k: list(data.X_lab[k]) + list(data.X_test[k]) for k in data.X_lab}
        g_all = None if groups is None else np.concatenate([groups, gt + groups.max() + 1])
        reps2, err = E.train_predict_bagged(task, X_all, y_all, data.X_test, gt, 200, bags, g_all)
        assert not err, err
        out2 = E.ensemble_output(task, reps2, weights)
    return mod, data, {"stage1": out1, "final": out2, "retrain_info": info}, len(y)


def main():
    src = dict(a.split("=") for a in sys.argv[1:])
    results, manifest = {}, {}
    for name in ("door", "acv", "rail", "shm"):
        report = json.loads((REPO / src[name] / f"report_{name}.json").read_text())
        E.log(f"[{name}] {report['mode']}: {report.get('program', report.get('program_types'))}")
        mod, data, outs, n_lab = predict_both(name, report)
        results[name] = (mod, data, outs)
        d = report["decision"]
        manifest[name] = {"mode": report["mode"], "program": report.get("program", report.get("program_types")),
                          "labelled_rows_used": n_lab, "best_single_oof": d["best_single_oof"],
                          "nested_ensemble_oof": d["nested_ensemble_oof"]}
    for folder, key in VARIANTS.items():
        out = REPO / "submissions" / folder
        out.mkdir(parents=True, exist_ok=True)
        for name, (mod, data, outs) in results.items():
            mod.write(data, {"final": outs[key]}, out)
        man = {k: dict(v) for k, v in manifest.items()}
        if key == "final":
            for name, (_, _, outs) in results.items():
                man[name]["hard_label_retraining"] = outs["retrain_info"]
        (out / "manifest.json").write_text(json.dumps({"variant": folder, "tasks": man}, indent=2, default=str) + "\n")
        subprocess.run([sys.executable, str(REPO / "scripts" / "make_submission.py"), "--skip-run", "--out",
                        f"submissions/{folder}"], check=True)
    for name, (_, data, outs) in results.items():
        a, b = outs["stage1"], outs["final"]
        if isinstance(a, np.ndarray) and a.dtype == object:
            print(f"[{name}] labels changed by retraining: {int((a != b).sum())} of {len(a)}")
        elif name == "shm":
            r = np.abs(b - a) / a
            print(f"[{name}] values changed by retraining: mean {r.mean()*100:.2f}%, max {r.max()*100:.2f}%")
        else:
            ra = [data.test_ids[i][1] for i in np.argsort(-a)]
            rb = [data.test_ids[i][1] for i in np.argsort(-b)]
            print(f"[{name}] ranking without: {'|'.join(ra)}  with: {'|'.join(rb)}  (top-1 same: {ra[0] == rb[0]})")


if __name__ == "__main__":
    main()
