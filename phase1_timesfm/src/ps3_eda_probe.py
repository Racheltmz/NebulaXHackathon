"""EDA on frozen TimesFM3 features -- statistics only, no modeling.

Goal: sanity-check the cached representation (finiteness, scale, class
balance) before spending any training epochs on it. No classical ML models
here (per direction: deep learning only) -- descriptive stats only.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", choices=["classification", "regression"], required=True)
    ap.add_argument("--features-root", default="data/ps3_features")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fs = Path(a.features_root) / a.task
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    feats = train["features"] + test["features"]
    labels = list(train["labels"]) + list(test["labels"])
    x = torch.stack([z.float() for z in feats])

    n_bad = int((~torch.isfinite(x)).sum())
    log(f"\n=== {a.task} EDA ===")
    log(f"n={x.shape[0]} channels={x.shape[1]} patches={x.shape[2]} dim={x.shape[3]}")
    log(f"non-finite raw values: {n_bad} ({100*n_bad/x.numel():.4f}%)")
    xc = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    log(f"value range after clamp: [{xc.min():.2f}, {xc.max():.2f}], abs-mean={xc.abs().mean():.3f}, "
        f"per-example std spread=[{xc.std((1,2,3)).min():.3f}, {xc.std((1,2,3)).max():.3f}]")

    summary = {"task": a.task, "n": len(labels), "channels": int(x.shape[1]),
               "non_finite_pct": 100 * n_bad / x.numel()}
    if a.mode == "classification":
        vals, counts = np.unique(labels, return_counts=True)
        log(f"classes: {dict(zip(vals.tolist(), counts.tolist()))}")
        summary["class_counts"] = {str(v): int(c) for v, c in zip(vals, counts)}
        summary["imbalance_ratio"] = float(counts.max() / counts.min())
    else:
        yy = np.asarray(labels, dtype=float)
        log(f"target range: [{yy.min():.3g}, {yy.max():.3g}] mean={yy.mean():.3g} std={yy.std():.3g}")
        summary["target_range"] = [float(yy.min()), float(yy.max())]
        summary["target_mean"] = float(yy.mean())
        summary["target_std"] = float(yy.std())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
