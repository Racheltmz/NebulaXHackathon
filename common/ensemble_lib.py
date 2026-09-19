"""Fold-scheme ensembles with out-of-fold weight calibration and pseudo-label retraining.

Pipeline (per task)
  1. TYPES     The best evolved classical programs of a track (from the checkpoint archives, plus seed programs).
               Weak types are pruned by a cheap all-data out-of-fold pass.
  2. MEMBERS   For each type and each fold SCHEME, several members are trained on different training subsets:
                 div3    every member sees ALL flagged (faulty) samples + a disjoint third of the non-flagged ones
                 ovl     every member sees all flagged samples + a random 2/3 of the non-flagged ones
                 kfold   leave-one-fold-out cross-fit members on stratified 5 folds (leave-one-CASE-out for ACV)
                 boot    5 stratified bootstrap resamples (within flagged / non-flagged separately)
                 sub80   5 stratified 80% subsamples without replacement
                 full    one member on all data
  3. CALIBRATE Weights over (type, scheme) keys are chosen on out-of-fold predictions of an outer CV (stratified
               5-fold; leave-one-case-out for ACV).  Structured candidates (uniform, one key, one scheme, one type)
               plus random Dirichlet weights are scored; among candidates within 1e-3 of the best OOF score the most
               uniform one wins.  A prior-shift / output-level correction is calibrated the same way (kept at 1.0
               unless the OOF evidence requires it).  Because every single key is a candidate, the calibrated OOF
               score is never below the best single program's.
  4. PREDICT   Train the members on ALL labelled data (test-time bagging over several random partitions) and predict
               the held-out set.
  5. HARD-LABEL + RETRAIN  The ensemble's held-out predictions become pseudo-labels (class / top-1 faulty car /
               predicted value); the members are retrained on labelled + pseudo-labelled data and predict again with the
               calibrated weights.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

ALL_SCHEMES = ["full", "div3", "ovl", "kfold", "boot", "sub80"]
DEFAULT_SCHEMES = ["full", "kfold", "ovl", "sub80", "div3"]   # `boot` was the weakest in scripts/compare_fold_schemes.py


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ members ----


@dataclass(frozen=True)
class MemberType:
    name: str
    code: str
    input_key: str


def load_member_types(task_dir: Path, kinds: dict, top_p: int) -> list:
    """`kinds` maps checkpoint-archive kind -> input key or (input key, top_p)."""
    out = []
    for kind, spec in kinds.items():
        input_key, p = spec if isinstance(spec, tuple) else (spec, top_p)
        path = Path(task_dir) / "checkpoints" / f"{kind}_top10.jsonl.gz"
        progs = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        seen = set()
        for prog in sorted(progs, key=lambda q: -q["metrics"]["combined_score"]):
            h = hashlib.md5(prog["code"].encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append(MemberType(f"{kind}:{prog['id'][:6]}", prog["code"], input_key))
            if len(seen) >= p:
                break
    return out


def _run_member(code: str, X_tr, y_tr, X_pred):
    warnings.simplefilter("ignore")
    ns = {"__name__": "member"}
    try:
        exec(compile(code, "<member>", "exec"), ns)
        model = ns["Model"]()
        model.fit(X_tr, y_tr)
        return np.asarray(model.predict(X_pred)), None
    except Exception as exc:  # noqa: BLE001 - a broken member is reported, not fatal
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"


# ------------------------------------------------------------- fold schemes ----


def scheme_subsets(scheme: str, flag: np.ndarray, groups, rng: np.random.Generator) -> list:
    """Training-index arrays (into the labelled set) for one fold scheme."""
    flag = np.asarray(flag, dtype=bool)
    n = len(flag)
    pos, neg = np.flatnonzero(flag), np.flatnonzero(~flag)
    if scheme == "full":
        return [np.arange(n)]
    if scheme == "div3":
        return [np.sort(np.concatenate([pos, part])) for part in np.array_split(rng.permutation(neg), 3)]
    if scheme == "ovl":
        k = max(1, int(round(len(neg) * 2 / 3)))
        return [np.sort(np.concatenate([pos, rng.choice(neg, k, replace=False)])) for _ in range(3)]
    if scheme == "kfold":
        if groups is not None:
            return [np.flatnonzero(groups != g) for g in np.unique(groups)]
        folds = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1 << 30))).split(np.zeros(n), flag)
        return [np.sort(tr) for tr, _ in folds]
    if scheme == "boot":
        return [np.sort(np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]))
                for _ in range(5)]
    if scheme == "sub80":
        kp, kn = max(1, int(round(len(pos) * .8))), max(1, int(round(len(neg) * .8)))
        return [np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, kn, replace=False)]))
                for _ in range(5)]
    raise ValueError(scheme)


# --------------------------------------------------------------------- task ----


@dataclass
class EnsembleTask:
    name: str
    kind: str                       # 'class' | 'rank' | 'reg'
    types: list
    metric: Callable                # (y_true, output, groups) -> float
    make_flag: Callable             # (y_lab) -> (y -> bool array); may capture a threshold from y_lab
    classes: list | None = None     # class kind: label order
    flagged_classes: list = field(default_factory=list)   # tie-break toward these
    schemes: list = field(default_factory=lambda: list(DEFAULT_SCHEMES))
    n_jobs: int = int(os.environ.get("ENSEMBLE_JOBS", "3"))
    flag_scale: float = 1.0         # class kind: prior-shift correction on flagged-class scores (calibrated)
    out_scale: float = 1.0          # reg kind: multiplicative output correction (calibrated)


def _subset(X, idx):
    return [X[i] for i in idx]


def _rep_from_members(task: EnsembleTask, outs: list, groups) -> np.ndarray:
    """Collapse one key's member outputs into a per-sample representation."""
    if task.kind == "class":
        idx = {c: i for i, c in enumerate(task.classes)}
        frac = np.zeros((len(outs[0]), len(task.classes)))
        for o in outs:
            for i, lab in enumerate(o):
                j = idx.get(str(lab))
                if j is not None:                    # an unexpected label simply casts no vote
                    frac[i, j] += 1.0 / len(outs)
        return frac
    if task.kind == "rank":
        zs = []
        for o in outs:
            o = np.asarray(o, dtype=float)
            z = np.zeros_like(o)
            for g in np.unique(groups):
                m = groups == g
                z[m] = (o[m] - o[m].mean()) / (o[m].std() + 1e-9)
            zs.append(z)
        return np.mean(zs, axis=0)
    return np.mean([np.log(np.maximum(np.asarray(o, dtype=float), 1e-6)) for o in outs], axis=0)


def train_predict(task: EnsembleTask, X_lab: dict, y: np.ndarray, X_pred: dict, groups_pred, seed: int,
                  groups_train=None, schemes=None, types=None):
    """Train every (type, scheme) member set on (X_lab, y); return {`type|scheme`: representation} and errors."""
    rng = np.random.default_rng(seed)
    flag = task.make_flag(y)(y)
    jobs, keys = [], []
    for scheme in (schemes or task.schemes):
        subsets = scheme_subsets(scheme, flag, groups_train, rng)
        for t in (types or task.types):
            for idx in subsets:
                jobs.append(delayed(_run_member)(t.code, _subset(X_lab[t.input_key], idx), y[idx], X_pred[t.input_key]))
                keys.append(f"{t.name}|{scheme}")
    results = Parallel(n_jobs=task.n_jobs)(jobs)
    per_key, errors = {}, []
    for key, (out, err) in zip(keys, results):
        if err:
            errors.append(f"{key}: {err}")
        else:
            per_key.setdefault(key, []).append(out)
    return {k: _rep_from_members(task, outs, groups_pred) for k, outs in per_key.items()}, sorted(set(errors))


def train_predict_bagged(task: EnsembleTask, X_lab: dict, y, X_pred: dict, groups_pred, seed: int, bags: int,
                         groups_train=None):
    """Test-time bagging: repeat the random schemes with `bags` different seeds and average each key's
    representation (`full` is deterministic and computed once)."""
    acc, errs = {}, []
    for b in range(bags):
        schemes = task.schemes if b == 0 else [s for s in task.schemes if s != "full"]
        if not schemes:
            break
        reps, e = train_predict(task, X_lab, y, X_pred, groups_pred, seed * 1000 + b, groups_train, schemes)
        errs += e
        for k, r in reps.items():
            acc.setdefault(k, []).append(r)
    return {k: np.mean(v, axis=0) for k, v in acc.items()}, sorted(set(errs))


def ensemble_output(task: EnsembleTask, reps: dict, weights: dict):
    names = [n for n in reps if weights.get(n, 0) > 0]
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    total = sum(wi * reps[n] for wi, n in zip(w, names))
    if task.kind == "class":
        scale = np.array([task.flag_scale if c in task.flagged_classes else 1.0 for c in task.classes])
        bonus = np.array([1e-9 if c in task.flagged_classes else 0.0 for c in task.classes])
        return np.array(task.classes, dtype=object)[np.argmax(total * scale + bonus, axis=1)]
    return total if task.kind == "rank" else np.exp(total) * task.out_scale


# --------------------------------------------------------------- calibration ----


def outer_splits(task: EnsembleTask, y, groups, n_outer: int, seed: int):
    flag = task.make_flag(y)(y)
    if task.kind == "rank":
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(np.zeros(len(y)), flag))


def oof_representations(task: EnsembleTask, X_lab: dict, y: np.ndarray, groups, n_outer: int, seed: int,
                        schemes=None, types=None):
    reps = {}
    splits = outer_splits(task, y, groups, n_outer, seed)
    for s, (tr, va) in enumerate(splits):
        log(f"  [{task.name}] outer split {s + 1}/{len(splits)}: train {len(tr)}, validate {len(va)}")
        Xtr = {k: _subset(v, tr) for k, v in X_lab.items()}
        Xva = {k: _subset(v, va) for k, v in X_lab.items()}
        r, errs = train_predict(task, Xtr, y[tr], Xva, None if groups is None else groups[va], seed + s,
                                None if groups is None else groups[tr], schemes, types)
        for e in errs:
            log(f"    member error: {e}")
        for name, rep in r.items():
            if name not in reps:
                reps[name] = np.full((len(y),) + rep.shape[1:], np.nan)
            reps[name][va] = rep
    return reps


def prune_types(task: EnsembleTask, X_lab, y, groups, n_outer, seed, keep: int = 3, margin: float = 0.06):
    """Cheap pass: all-data OOF per type; keep the best `keep` types within `margin` of the best."""
    reps = oof_representations(task, X_lab, y, groups, n_outer, seed, schemes=["full"])
    scores = {}
    for t in task.types:
        key = f"{t.name}|full"
        if key in reps and not np.isnan(reps[key]).any():
            scores[t.name] = float(task.metric(y, ensemble_output(task, {key: reps[key]}, {key: 1.0}), groups))
    best = max(scores.values())
    ranked = sorted(scores, key=lambda n: -scores[n])
    kept = [n for n in ranked if scores[n] >= best - margin][:keep]
    task.types = [t for t in task.types if t.name in kept]
    return scores, kept


def _structured_candidates(names):
    T = len(names)
    schemes = sorted({n.split("|")[1] for n in names})
    types = sorted({n.split("|")[0] for n in names})
    cands = [np.ones(T) / T]
    cands += [np.eye(T)[i] for i in range(T)]
    for s in schemes:
        v = np.array([1.0 if n.endswith("|" + s) else 0.0 for n in names])
        cands.append(v / v.sum())
    for t in types:
        v = np.array([1.0 if n.startswith(t + "|") else 0.0 for n in names])
        cands.append(v / v.sum())
    return cands


def calibrate_weights(task: EnsembleTask, reps: dict, y: np.ndarray, groups, seed: int = 0):
    names = [n for n in reps if not np.isnan(reps[n]).any()]
    rng = np.random.default_rng(seed)
    T = len(names)

    def score(w):
        return float(task.metric(y, ensemble_output(task, {n: reps[n] for n in names}, dict(zip(names, w))), groups))

    struct = _structured_candidates(names)
    cands = struct + list(rng.dirichlet(np.ones(T), 2000)) + list(rng.dirichlet(np.full(T, 0.3), 1000))
    scores = np.array([score(w) for w in cands])
    best = scores.max()
    eligible = np.flatnonzero(scores >= best - 1e-3)
    ent = lambda w: float(-(w[w > 0] * np.log(w[w > 0])).sum())
    pick = max(eligible, key=lambda i: ent(cands[i]))
    weights = dict(zip(names, map(float, cands[pick])))
    diag = {
        "single_key_oof": {n: float(scores[1 + i]) for i, n in enumerate(names)},
        "uniform_oof": float(scores[0]),
        "best_oof": float(best),
        "chosen_oof": float(scores[pick]),
        "n_structured_candidates": len(struct),
        "note": "chosen_oof is optimistic: the weights were fitted on these same OOF predictions",
    }
    return weights, diag


def nested_ensemble_score(task: EnsembleTask, reps: dict, y, groups, n_outer: int, seed: int):
    """Honest OOF score of the calibrate-then-ensemble recipe: every outer fold is scored with weights fitted
    only on the OTHER folds' out-of-fold predictions (no extra model training needed)."""
    names = [n for n in reps if not np.isnan(reps[n]).any()]
    dt = object if task.kind == "class" else float
    pooled = np.empty(len(y), dtype=dt)
    for tr, va in outer_splits(task, y, groups, n_outer, seed):
        w, _ = calibrate_weights(task, {n: reps[n][tr] for n in names}, y[tr], None if groups is None else groups[tr], seed)
        pooled[va] = ensemble_output(task, {n: reps[n][va] for n in names}, w)
    return float(task.metric(y, pooled, groups))


def calibrate_scalar(task: EnsembleTask, attr: str, grid, reps: dict, weights: dict, y, groups):
    """Correction closest to 1.0 (log scale) among values within 1e-3 of the best OOF score."""
    scores = []
    for v in grid:
        setattr(task, attr, float(v))
        scores.append(float(task.metric(y, ensemble_output(task, reps, weights), groups)))
    scores = np.array(scores)
    eligible = np.flatnonzero(scores >= scores.max() - 1e-3)
    pick = min(eligible, key=lambda i: abs(np.log(grid[i])))
    setattr(task, attr, float(grid[pick]))
    return {"value": float(grid[pick]), "oof_at_value": float(scores[pick]),
            "oof_at_1.0": float(scores[int(np.argmin(np.abs(np.log(grid))))]), "oof_best": float(scores.max())}


# ---------------------------------------------------------------- pseudo-labels ----


def pseudo_labels(task: EnsembleTask, out: np.ndarray, groups_test):
    if task.kind == "rank":
        y = np.zeros(len(out), dtype=int)
        for g in np.unique(groups_test):
            idx = np.flatnonzero(groups_test == g)
            y[idx[np.argmax(out[idx])]] = 1
        return y
    return out.astype(object) if task.kind == "class" else out.astype(float)


def _change_summary(task: EnsembleTask, a, b, groups_test):
    if task.kind == "class":
        return {"fraction_changed": float(np.mean(np.asarray(a) != np.asarray(b))), "n": int(len(a))}
    if task.kind == "rank":
        return {"top1_changed_per_group": [bool(np.argmax(a[groups_test == g]) != np.argmax(b[groups_test == g]))
                                           for g in np.unique(groups_test)]}
    rel = np.abs(np.asarray(b) - np.asarray(a)) / np.maximum(np.abs(np.asarray(a)), 1e-9)
    return {"mean_rel_change": float(rel.mean()), "max_rel_change": float(rel.max()), "n": int(len(a))}


# ------------------------------------------------------------------- pipeline ----


def run(task: EnsembleTask, data, n_outer: int = 5, seed: int = 0, bags: int = 3, prune: bool = True,
        keep_types: int = 3, retrain: bool = True) -> dict:
    t0 = time.time()
    y = np.asarray(data.y_lab)
    groups = data.groups_lab
    log(f"[{task.name}] {len(y)} labelled, {len(data.test_ids)} held-out; {len(task.types)} candidate types; "
        f"schemes {task.schemes}; flagged={int(task.make_flag(y)(y).sum())}")
    pruning = None
    if prune and len(task.types) > 1:
        log(f"[{task.name}] pruning weak program types (all-data OOF)")
        scores, kept = prune_types(task, data.X_lab, y, groups, n_outer, seed, keep=keep_types)
        pruning = {"type_oof": scores, "kept": kept}
        log(f"[{task.name}] type OOF {json.dumps({k: round(v, 4) for k, v in scores.items()})} -> kept {kept}")
    log(f"[{task.name}] OOF weight calibration over {len(task.types)} types x {len(task.schemes)} schemes")
    oof = oof_representations(task, data.X_lab, y, groups, n_outer, seed)
    weights, diag = calibrate_weights(task, oof, y, groups, seed)
    scalar_diag = None
    if task.kind in ("class", "reg"):
        attr, grid = (("flag_scale", np.round(np.arange(0.30, 1.51, 0.05), 2)) if task.kind == "class"
                      else ("out_scale", np.round(np.arange(0.80, 1.201, 0.02), 2)))
        for _ in range(2):
            scalar_diag = calibrate_scalar(task, attr, grid, {n: oof[n] for n in weights}, weights, y, groups)
            weights, diag = calibrate_weights(task, oof, y, groups, seed)
        scalar_diag["attr"] = attr
    log(f"[{task.name}] weights {json.dumps({k: round(v, 3) for k, v in weights.items() if v > 0.01})}  "
        f"OOF {diag['chosen_oof']:.4f} (uniform {diag['uniform_oof']:.4f}, best single key {max(diag['single_key_oof'].values()):.4f})")

    # --- ensemble vs the single fully-trained program (no out-of-fold on the latter's final fit) ---------------
    full_keys = [k for k in oof if k.endswith("|full") and not np.isnan(oof[k]).any()]
    single_scores = {k: float(task.metric(y, ensemble_output(task, {k: oof[k]}, {k: 1.0}), groups)) for k in full_keys}
    best_key = max(single_scores, key=single_scores.get) if single_scores else None
    nested = nested_ensemble_score(task, oof, y, groups, n_outer, seed)
    use_ensemble = best_key is None or nested > single_scores[best_key] + 1e-9
    decision = {"nested_ensemble_oof": nested, "best_single_key": best_key,
                "best_single_oof": single_scores.get(best_key), "use_ensemble": bool(use_ensemble),
                "rule": "ensemble is used only if its nested OOF score is strictly higher than the best single program "
                        "trained on all data; otherwise that single fully-trained program is used"}
    log(f"[{task.name}] nested-honest ensemble OOF {nested:.4f} vs best single fully-trained "
        f"{single_scores.get(best_key, float('nan')):.4f} ({best_key}) -> {'ENSEMBLE' if use_ensemble else 'SINGLE fully-trained program'}")

    err1, err2 = [], []
    if not use_ensemble:
        task.flag_scale, task.out_scale = 1.0, 1.0
        single_type = [t for t in task.types if f"{t.name}|full" == best_key]
        reps1, err1 = train_predict(task, data.X_lab, y, data.X_test, data.groups_test, seed + 100, groups, ["full"], single_type)
        out1 = ensemble_output(task, reps1, {best_key: 1.0})
        report = {
            "task": task.name, "n_labelled": int(len(y)), "n_heldout": int(len(data.test_ids)), "mode": "single fully-trained program",
            "program": best_key.split("|")[0], "schemes": task.schemes, "program_types": [t.name for t in task.types],
            "pruning": pruning, "decision": decision, "ensemble_candidate": {"weights": weights, "calibration": diag},
            "member_errors": err1, "seconds": round(time.time() - t0, 1),
        }
        log(f"[{task.name}] done in {report['seconds']}s (single fully-trained program)")
        return {"stage1": out1, "final": out1, "weights": {best_key: 1.0}, "report": report}

    log(f"[{task.name}] train on all labelled data, predict held-out")
    reps1, err1 = train_predict_bagged(task, data.X_lab, y, data.X_test, data.groups_test, seed + 100, bags, groups)
    out1 = ensemble_output(task, reps1, weights)
    out2 = out1
    if retrain:
        log(f"[{task.name}] hard-label held-out set and retrain")
        y_pseudo = pseudo_labels(task, out1, data.groups_test)
        y_all = np.concatenate([y.astype(object), y_pseudo]) if task.kind == "class" else np.concatenate([y, y_pseudo])
        X_all = {k: list(data.X_lab[k]) + list(data.X_test[k]) for k in data.X_lab}
        g_all = None if groups is None else np.concatenate([groups, data.groups_test + groups.max() + 1])
        reps2, err2 = train_predict_bagged(task, X_all, y_all, data.X_test, data.groups_test, seed + 200, bags, g_all)
        out2 = ensemble_output(task, reps2, weights)
    report = {
        "task": task.name, "n_labelled": int(len(y)), "n_heldout": int(len(data.test_ids)),
        "mode": "ensemble", "schemes": task.schemes, "program_types": [t.name for t in task.types], "pruning": pruning,
        "bags": bags, "decision": decision, "weights": weights, "calibration": diag, "scalar_correction": scalar_diag,
        "flag_scale": task.flag_scale, "out_scale": task.out_scale,
        "member_errors": sorted(set(err1 + err2)),
        "retrain_change_vs_stage1": _change_summary(task, out1, out2, data.groups_test) if retrain else None,
        "seconds": round(time.time() - t0, 1),
    }
    log(f"[{task.name}] done in {report['seconds']}s; change after retraining: {report['retrain_change_vs_stage1']}")
    return {"stage1": out1, "final": out2, "weights": weights, "report": report}


def cli(name: str, loader: Callable, build: Callable, write: Callable, default_top_p: int = 3,
        default_schemes: list | None = None):
    """Shared entry point for the per-task ensemble scripts."""
    import argparse
    ap = argparse.ArgumentParser(description=f"{name}: fold-scheme ensemble + hard-label retraining")
    ap.add_argument("--out", default="submission", help="output directory")
    ap.add_argument("--top-p", type=int, default=default_top_p, help="program types taken per archive kind")
    ap.add_argument("--schemes", nargs="+", default=default_schemes, choices=ALL_SCHEMES)
    ap.add_argument("--bags", type=int, default=3, help="random partitions averaged at test time")
    ap.add_argument("--outer", type=int, default=5, help="outer CV folds for weight calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-retrain", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = loader()
    task = build(data, a.top_p)
    if a.schemes:
        task.schemes = a.schemes
    res = run(task, data, n_outer=a.outer, seed=a.seed, bags=a.bags, retrain=not a.no_retrain)
    path = write(data, res, out)
    (out / f"report_{name}.json").write_text(json.dumps(res["report"], indent=2, default=str) + "\n")
    log(f"[{name}] wrote {path}")
    return res
