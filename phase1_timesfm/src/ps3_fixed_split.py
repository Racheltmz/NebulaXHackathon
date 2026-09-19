"""Train ONLY on the original train.pt, evaluate ONLY on the original test.pt.

Purpose: the pooled 3-fold CV used everywhere else mixes train+test into one
pool and re-splits it, which is standard practice for a small-data estimate
but can look optimistic if the original train/test split carries a real
distribution shift (confirmed via ps3_eda2.py: normalized shift 22.5/12.3/15.3
for door/rail/acv). This script instead respects the original split exactly
once, matching what a real "train on known data, predict on genuinely held
out data" evaluation looks like -- the closest local proxy to the actual
competition grading setup.

Also supports --drop-channels (indices to zero out before pooling) and
--keep-channels (indices to keep, all others dropped) so "are all channels
actually load-bearing, or can we prune the weak ones" can be tested directly
against held-out data rather than assumed from the Cohen's d ranking alone.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_percentage_error

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ps3_dl import Model, mask_patches, mixup, log


def apply_channel_filter(x, keep_channels=None, drop_channels=None):
    if keep_channels is None and drop_channels is None:
        return x
    c = x.shape[1]
    mask = torch.ones(c, dtype=torch.bool)
    if keep_channels is not None:
        mask[:] = False
        mask[keep_channels] = True
    if drop_channels is not None:
        mask[drop_channels] = False
    return x[:, mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--arch", default="mil")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mask-p", type=float, default=0.15)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--mixup-alpha", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--keep-channels", default=None, help="comma-separated indices to keep")
    ap.add_argument("--drop-channels", default=None, help="comma-separated indices to drop")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = Path(args.features)
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    xtr = torch.nan_to_num(torch.stack([z.float() for z in train["features"]]),
                            nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    xte = torch.nan_to_num(torch.stack([z.float() for z in test["features"]]),
                            nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

    keep = [int(i) for i in args.keep_channels.split(",")] if args.keep_channels else None
    drop = [int(i) for i in args.drop_channels.split(",")] if args.drop_channels else None
    xtr = apply_channel_filter(xtr, keep, drop)
    xte = apply_channel_filter(xte, keep, drop)
    log(f"n_train={len(xtr)} n_test={len(xte)} channels_used={xtr.shape[1]} arch={args.arch}")

    dim = xtr.shape[-1]
    if args.task == "classification":
        vals = sorted(set(train["labels"]) | set(test["labels"]))
        mp = {v: i for i, v in enumerate(vals)}
        ytr = torch.tensor([mp[v] for v in train["labels"]])
        yte = torch.tensor([mp[v] for v in test["labels"]])
        n_out = len(vals)
        counts = torch.bincount(ytr, minlength=n_out).float().clamp_min(1)
        class_weight = counts.sum() / (n_out * counts)
        log(f"classes={vals} train_counts={counts.tolist()} class_weight={class_weight.tolist()}")
    else:
        ytr = torch.tensor([float(v) for v in train["labels"]])
        yte = torch.tensor([float(v) for v in test["labels"]])
        n_out = 1

    torch.manual_seed(args.seed)
    model = Model(args.arch, dim, n_out, hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xtr_d, xte_d = xtr.to(dev), xte.to(dev)

    with torch.no_grad():
        pooled_tr_raw = model.pool(xtr_d)
        mean, std = pooled_tr_raw.mean(0, keepdim=True), pooled_tr_raw.std(0, keepdim=True).clamp_min(1e-4)

    if args.task == "classification":
        ytr_dev = ytr.to(dev)
        y_onehot = torch.nn.functional.one_hot(ytr_dev, n_out).float()
        cw = class_weight.to(dev)
        sample_w_all = cw[ytr_dev]
    else:
        y_log = torch.log(ytr.clamp_min(1e-5))
        y_mean, y_std = y_log.mean(), y_log.std().clamp_min(1e-4)
        ytr_dev = ((y_log - y_mean) / y_std).to(dev).unsqueeze(-1)
        sample_w_all = torch.ones(len(ytr), device=dev)

    n = xtr_d.shape[0]
    bs = min(args.batch_size, max(2, n))
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = mask_patches(xtr_d[idx], args.mask_p)
            xb = xb + args.noise_std * torch.randn_like(xb)
            pooled = (model.pool(xb) - mean) / std
            if args.task == "classification":
                pooled, target, sw = mixup(pooled, y_onehot[idx], sample_w_all[idx], args.mixup_alpha)
                logits = model(pooled)
                logp = torch.log_softmax(logits, -1)
                smooth = args.label_smoothing
                target = target * (1 - smooth) + smooth / n_out
                loss = -(sw * (target * logp).sum(-1)).mean()
            else:
                pooled, target, _ = mixup(pooled, ytr_dev[idx], sample_w_all[idx], args.mixup_alpha)
                pred = model(pooled)
                loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pooled_te = (model.pool(xte_d) - mean) / std
        if args.task == "classification":
            pred = model(pooled_te).argmax(-1).cpu()
            metric = {"balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                      "macro_f1": float(f1_score(yte, pred, average="macro"))}
        else:
            pred = torch.exp(model(pooled_te).squeeze(-1) * y_std + y_mean).cpu()
            yv, pv = yte.clamp_min(1e-6), pred.clamp_min(1e-6)
            m = float(mean_absolute_percentage_error(yv, pv))
            metric = {"mape": m, "score": max(0.0, 1 - m)}

    result = {"task": fs.name, "arch": args.arch, "n_train": len(xtr), "n_test": len(xte),
              "channels_used": xtr.shape[1], "keep_channels": keep, "drop_channels": drop,
              "metric": metric, "elapsed": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    log(f"FIXED-SPLIT RESULT: {json.dumps(metric)}")


if __name__ == "__main__":
    main()
