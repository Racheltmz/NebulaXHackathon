"""Door/Rail classification with raw-signal statistics fused into the pooled
embedding, extending the SHM raw-stat fusion idea (which took SHM's score
from 0.300 to 0.456) to the classification tasks.

Per-channel raw stats (rms, peak, std, mean_abs, plus range=max-min) are
computed for every channel and concatenated as extra scalar features
alongside the pooled TimesFM3 embedding, then both feed the same MLP head.
3-fold stratified CV, same protocol as ps3_dl.py, for direct comparability.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ps3_dl import ChannelAttnPool, MILAttnPool, mask_patches, mixup, log


def raw_stats_per_example(x_list):
    rows = []
    for ex in x_list:
        ex = np.nan_to_num(np.asarray(ex, dtype=np.float64))
        rms = np.sqrt((ex ** 2).mean(-1))
        peak = np.abs(ex).max(-1)
        std = ex.std(-1)
        mean_abs = np.abs(ex).mean(-1)
        rng = ex.max(-1) - ex.min(-1)
        # aggregate across channels (mean+max) so the stat vector has a fixed
        # size regardless of how many raw channels this task has
        agg = np.concatenate([
            [rms.mean(), rms.max(), peak.mean(), peak.max(), std.mean(), std.max(),
             mean_abs.mean(), mean_abs.max(), rng.mean(), rng.max()]
        ])
        rows.append(np.log1p(np.abs(agg)) * np.sign(agg))
    return np.array(rows, dtype=np.float32)


class FusionModel(nn.Module):
    def __init__(self, dim, n_stats, n_out, backbone="mil", hidden=256, dropout=0.3):
        super().__init__()
        self.pool = MILAttnPool(dim) if backbone == "mil" else ChannelAttnPool(dim)
        feat_dim = 2 * dim + n_stats
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, tokens, stats):
        pooled = self.pool(tokens)
        return self.head(torch.cat([pooled, stats], -1)), pooled


def run_fold(xtr, str_, ytr, xte, ste_, n_out, args, dev, seed, class_weight):
    torch.manual_seed(seed)
    dim = xtr.shape[-1]
    model = FusionModel(dim, str_.shape[1], n_out, args.backbone, args.hidden, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xtr, xte = xtr.to(dev), xte.to(dev)

    with torch.no_grad():
        pooled_raw = model.pool(xtr)
        p_mean, p_std = pooled_raw.mean(0, keepdim=True), pooled_raw.std(0, keepdim=True).clamp_min(1e-4)
    s_mean, s_std = str_.mean(0, keepdim=True), str_.std(0, keepdim=True).clamp_min(1e-4)
    str_n = ((str_ - s_mean) / s_std).to(dev)
    ste_n = ((ste_ - s_mean) / s_std).to(dev)

    ytr_dev = ytr.to(dev)
    y_onehot = torch.nn.functional.one_hot(ytr_dev, n_out).float()
    cw = class_weight.to(dev)
    sample_w_all = cw[ytr_dev]

    n = xtr.shape[0]
    bs = min(args.batch_size, max(2, n))
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = mask_patches(xtr[idx], args.mask_p)
            xb = xb + args.noise_std * torch.randn_like(xb)
            logits, _ = model(xb, str_n[idx])
            pooled_dummy = None
            target = y_onehot[idx]
            sw = sample_w_all[idx]
            logp = torch.log_softmax(logits, -1)
            smooth = args.label_smoothing
            target = target * (1 - smooth) + smooth / n_out
            loss = -(sw * (target * logp).sum(-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(xte, ste_n)
        pred = logits.argmax(-1).cpu()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--arrays-root", default="data/ps3_arrays")
    ap.add_argument("--task-name", required=True, help="door or rail (for raw array filenames)")
    ap.add_argument("--backbone", choices=["linear", "mil"], default="mil")
    ap.add_argument("--folds", type=int, default=3)
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
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = Path(args.features)
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    labels = list(train["labels"]) + list(test["labels"])
    x = torch.stack([z.float() for z in (train["features"] + test["features"])])
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    vals = sorted(set(labels)); mp = {v: i for i, v in enumerate(vals)}
    y = torch.tensor([mp[v] for v in labels])
    n_out = len(vals)

    raw_tr = np.load(Path(args.arrays_root) / f"{args.task_name}_train_x.npy", allow_pickle=True)
    raw_te = np.load(Path(args.arrays_root) / f"{args.task_name}_test_x.npy", allow_pickle=True)
    stats = torch.from_numpy(raw_stats_per_example(list(raw_tr) + list(raw_te)))
    log(f"n={len(y)} classes={vals} stats_dim={stats.shape[1]} backbone={args.backbone}")

    counts = np.bincount(y.numpy())
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    all_pred = torch.zeros_like(y)
    per_fold = []
    t0 = time.time()
    for i, (tr_idx, te_idx) in enumerate(splitter.split(np.zeros(len(y)), y.numpy())):
        ytr = y[tr_idx]
        cnt = torch.bincount(ytr, minlength=n_out).float().clamp_min(1)
        class_weight = cnt.sum() / (n_out * cnt)
        pred = run_fold(x[tr_idx], stats[tr_idx], ytr, x[te_idx], stats[te_idx], n_out, args, dev,
                         args.seed + i, class_weight)
        all_pred[te_idx] = pred
        fm = {"balanced_accuracy": float(balanced_accuracy_score(y[te_idx], pred)),
              "macro_f1": float(f1_score(y[te_idx], pred, average="macro"))}
        per_fold.append(fm)
        log(f"fold {i+1} done: {fm} (elapsed {time.time()-t0:.1f}s)")

    result = {"task": fs.name, "backbone": args.backbone, "n": len(y),
              "pooled": {"balanced_accuracy": float(balanced_accuracy_score(y, all_pred)),
                         "macro_f1": float(f1_score(y, all_pred, average="macro")),
                         "class_counts": {str(v): int(c) for v, c in zip(vals, counts)}},
              "per_fold": per_fold}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    log("POOLED RESULT (fusion): " + json.dumps(result["pooled"]))


if __name__ == "__main__":
    main()
