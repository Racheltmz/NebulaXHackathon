"""Follow-up EDA: resolve the rail 5-vs-17-channel mismatch, per-channel/feature
discriminative power for door/rail/shm, and train/test distribution shift.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


def rail_provenance():
    d_train = torch.load("data/ps3_features/rail/train.pt", weights_only=False)
    log(f"cached rail train.pt keys: {list(d_train.keys())}")
    log(f"declared 'channels' field: {d_train.get('channels')}")
    log(f"features[0].shape: {tuple(d_train['features'][0].shape)}")
    log(f"n examples: {len(d_train['features'])}")
    raw = np.load("data/ps3_arrays/rail_train_x.npy", allow_pickle=True)
    log(f"raw ps3_arrays rail_train_x[0].shape: {np.asarray(raw[0]).shape}, n={len(raw)}")
    log(f"raw labels vs cached labels first 5 match: "
        f"{np.load('data/ps3_arrays/rail_train_y.npy', allow_pickle=True)[:5].tolist()} vs "
        f"{d_train['labels'][:5]}")


def per_channel_discriminability(task, mode):
    fs = Path("data/ps3_features") / task
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    feats = train["features"] + test["features"]
    labels = list(train["labels"]) + list(test["labels"])
    x = torch.stack([z.float() for z in feats])
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    chan_vec = x.mean(2).norm(dim=-1).numpy()  # [N, C] per-channel embedding norm as a cheap summary

    log(f"\n=== {task}: per-channel discriminative power (Cohen's d vs rest, on channel-norm) ===")
    if mode == "classification":
        vals = sorted(set(labels))
        y = np.array([vals.index(v) for v in labels])
        if len(vals) == 2:
            g0, g1 = chan_vec[y == 0], chan_vec[y == 1]
            pooled_std = np.sqrt((g0.var(0) + g1.var(0)) / 2) + 1e-6
            d = (g0.mean(0) - g1.mean(0)) / pooled_std
            order = np.argsort(-np.abs(d))
            for c in order[:8]:
                log(f"  channel {c}: |Cohen's d|={abs(d[c]):.3f}")
            log(f"  max |d|={np.abs(d).max():.3f}, median |d|={np.median(np.abs(d)):.3f}")
        else:
            # one-vs-rest max d across classes
            maxd = np.zeros(chan_vec.shape[1])
            for v in range(len(vals)):
                g0, g1 = chan_vec[y == v], chan_vec[y != v]
                if len(g0) < 2 or len(g1) < 2:
                    continue
                pooled_std = np.sqrt((g0.var(0) + g1.var(0)) / 2) + 1e-6
                d = np.abs(g0.mean(0) - g1.mean(0)) / pooled_std
                maxd = np.maximum(maxd, d)
            order = np.argsort(-maxd)
            for c in order[:8]:
                log(f"  channel {c}: max one-vs-rest |Cohen's d|={maxd[c]:.3f}")
            log(f"  max |d|={maxd.max():.3f}, median |d|={np.median(maxd):.3f}")
    return chan_vec, labels


def train_test_shift(task, mode):
    fs = Path("data/ps3_features") / task
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    xtr = torch.stack([z.float() for z in train["features"]])
    xte = torch.stack([z.float() for z in test["features"]])
    xtr = torch.nan_to_num(xtr, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    xte = torch.nan_to_num(xte, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    ptr, pte = xtr.mean((1, 2)), xte.mean((1, 2))  # pooled per-example vector
    mtr, mte = ptr.mean(0), pte.mean(0)
    shift = (mtr - mte).norm().item() / (ptr.std(0).mean().item() + 1e-6)
    log(f"\n=== {task}: train/test distribution shift (pooled mean diff / pooled std) = {shift:.3f} ===")
    if mode == "classification":
        tr_labels = train["labels"]; te_labels = test["labels"]
        vals = sorted(set(tr_labels) | set(te_labels))
        tr_dist = {v: tr_labels.count(v) for v in vals} if isinstance(tr_labels, list) else None
        log(f"  train class counts: {[(v, list(tr_labels).count(v)) for v in vals]}")
        log(f"  test  class counts: {[(v, list(te_labels).count(v)) for v in vals]}")


def raw_stat_target_corr(task):
    """For regression (SHM): does a trivial raw-signal statistic already
    correlate with the target? If yes, that's a concrete, cheap auxiliary
    feature worth fusing into the head."""
    xtr = np.load(f"data/ps3_arrays/{task}_train_x.npy", allow_pickle=True)
    ytr = np.load(f"data/ps3_arrays/{task}_train_y.npy", allow_pickle=True).astype(float)
    xte = np.load(f"data/ps3_arrays/{task}_test_x.npy", allow_pickle=True)
    yte = np.load(f"data/ps3_arrays/{task}_test_y.npy", allow_pickle=True).astype(float)
    x_all = list(xtr) + list(xte)
    y_all = np.concatenate([ytr, yte])
    stats = {"rms": [], "peak": [], "std": [], "duration": [], "mean_abs": []}
    for ex in x_all:
        ex = np.nan_to_num(np.asarray(ex, dtype=np.float64))
        flat = ex.reshape(-1)
        stats["rms"].append(np.sqrt((flat ** 2).mean()))
        stats["peak"].append(np.abs(flat).max())
        stats["std"].append(flat.std())
        stats["duration"].append(ex.shape[-1])
        stats["mean_abs"].append(np.abs(flat).mean())
    log(f"\n=== {task}: raw-signal statistic vs target correlation (n={len(y_all)}) ===")
    for name, vals in stats.items():
        r = np.corrcoef(vals, y_all)[0, 1]
        log(f"  corr(target, {name}) = {r:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log("########## RAIL CHANNEL PROVENANCE ##########")
    rail_provenance()

    per_channel_discriminability("door", "classification")
    per_channel_discriminability("rail", "classification")
    per_channel_discriminability("acv", "classification")

    train_test_shift("door", "classification")
    train_test_shift("rail", "classification")
    train_test_shift("acv", "classification")

    raw_stat_target_corr("shm")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"status": "see stdout log"}, open(a.out, "w"))


if __name__ == "__main__":
    main()
