"""Self-contained linear classifier / regressor for PS3 on frozen TimesFM3 features.

Design (deliberately simple, FlaMinGo-style "linear probe on frozen backbone",
extended to be multi-channel-native):

  features [B, C, L, D]  (batch, channels, patches, transformer dim)
      -> per-channel mean+max pool over patches      -> [B, C, 2D]
      -> learned softmax channel attention pool       -> [B, 2D]
      -> per-feature z-score normalization (train-fold stats)
      -> single Linear(2D, n_out)                     (logistic/linear regression)
         (optional --hidden adds one GELU hidden layer; off by default)

Channel attention pooling is what makes this "multi-channel compatible":
Rail has 5 channels, Door/SHM/ACV have 1-8; the same head handles all of them
without any dataset-specific reshaping, and the pooling weights are learned
per-dataset rather than fixed to a channel count.

Robust low-data training mechanisms (all active by default, all ablatable):
  - feature masking: randomly zeroes whole (channel, patch) tokens each step
    (structured dropout on the *frozen representation*, not just the head),
    forcing the head to not rely on any single token.
  - Gaussian noise injected on the pooled, normalized feature vector.
  - mixup: linear-interpolates pairs of pooled features and their (soft)
    targets, which is the standard low-sample-size regularizer since it
    synthesizes new training points between real ones.
  - label smoothing (classification) / SmoothL1 on log-target (regression).
  - weight decay.

Evaluation: stratified K-fold CV for classification (KFold for regression),
reporting balanced accuracy + macro-F1 (classification) and the official
max(0, 1 - MAPE) score (regression), both per-fold and pooled across all
held-out predictions. Progress is logged every epoch with elapsed time and
an ETA so long runs are legible from the SLURM log.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_percentage_error


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LinearProbe(nn.Module):
    def __init__(self, dim, n_out, hidden=0, dropout=0.1):
        super().__init__()
        self.channel_score = nn.Linear(dim, 1)
        feat_dim = 2 * dim
        self.norm = nn.LayerNorm(feat_dim)
        if hidden > 0:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, n_out))

    def pool(self, features):
        # features: [B, C, L, D]
        mean_p = features.mean(2)             # [B, C, D]
        max_p = features.amax(2)              # [B, C, D]
        pooled_ch = torch.cat([mean_p, max_p], -1)  # [B, C, 2D]
        weights = torch.softmax(self.channel_score(mean_p), dim=1)  # [B, C, 1]
        return (pooled_ch * weights).sum(1)   # [B, 2D]

    def forward(self, pooled_norm):
        return self.head(pooled_norm)


def mask_patches(features, p):
    if p <= 0:
        return features
    keep = (torch.rand(features.shape[:3], device=features.device) > p).float().unsqueeze(-1)
    return features * keep / max(1e-3, 1 - p)


def mixup(x, y_onehot, alpha):
    if alpha <= 0:
        return x, y_onehot
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]


def run_fold(xtr, ytr, xte, yte, task, dim, n_out, args, dev, seed, fold_tag):
    torch.manual_seed(seed)
    model = LinearProbe(dim, n_out, hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xtr, xte = xtr.to(dev), xte.to(dev)

    with torch.no_grad():
        pooled_tr_raw = model.pool(xtr)
        mean, std = pooled_tr_raw.mean(0, keepdim=True), pooled_tr_raw.std(0, keepdim=True).clamp_min(1e-4)

    if task == "classification":
        ytr_dev, yte_dev = ytr.to(dev), yte.to(dev)
        y_onehot = torch.nn.functional.one_hot(ytr_dev, n_out).float()
    else:
        y_log = torch.log(ytr.clamp_min(1e-5))
        y_mean, y_std = y_log.mean(), y_log.std().clamp_min(1e-4)
        ytr_dev = ((y_log - y_mean) / y_std).to(dev).unsqueeze(-1)
        yte_dev = yte.to(dev)

    n = xtr.shape[0]
    bs = min(args.batch_size, max(2, n))
    steps_per_epoch = max(1, (n + bs - 1) // bs)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        ep_loss = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = mask_patches(xtr[idx], args.mask_p)
            xb = xb + args.noise_std * torch.randn_like(xb)
            pooled = (model.pool(xb) - mean) / std
            if task == "classification":
                pooled, target = mixup(pooled, y_onehot[idx], args.mixup_alpha)
                logits = model(pooled)
                logp = torch.log_softmax(logits, -1)
                smooth = args.label_smoothing
                target = target * (1 - smooth) + smooth / n_out
                loss = -(target * logp).sum(-1).mean()
            else:
                pooled, target = mixup(pooled, ytr_dev[idx], args.mixup_alpha)
                pred = model(pooled)
                loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.detach()) / steps_per_epoch
        if args.log_every and (ep + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            per_ep = elapsed / (ep + 1)
            eta = per_ep * (args.epochs - ep - 1)
            log(f"  fold {fold_tag} epoch {ep+1}/{args.epochs} loss={ep_loss:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    model.eval()
    with torch.no_grad():
        pooled_te = (model.pool(xte) - mean) / std
        if task == "classification":
            pred = model(pooled_te).argmax(-1).cpu()
        else:
            pred = torch.exp(model(pooled_te).squeeze(-1) * y_std + y_mean).cpu()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir with train.pt/test.pt from src/extract.py")
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=0, help="0 = pure linear/logistic regression")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mask-p", type=float, default=0.2)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--mixup-alpha", type=float, default=0.4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} task={args.task} features={args.features}")

    fs = Path(args.features)
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    feats = train["features"] + test["features"]
    labels = list(train["labels"]) + list(test["labels"])
    x = torch.stack([z.float() for z in feats])
    # extract.py stores features as float16; a handful of ACV's compressed
    # high-channel distribution tokens (variance across >8 channels) can
    # overflow fp16 range, producing inf that poisons every loss afterward.
    n_bad = int((~torch.isfinite(x)).sum())
    if n_bad:
        log(f"WARNING: {n_bad} non-finite feature values, clamping")
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    dim = x.shape[-1]
    log(f"n={len(labels)} channels={x.shape[1]} patches={x.shape[2]} dim={dim}")

    result = {"task": fs.name, "n": len(labels), "channels": int(x.shape[1]), "folds": args.folds,
              "hyperparams": dict(vars(args))}
    del result["hyperparams"]["features"], result["hyperparams"]["out"]

    if args.task == "classification":
        vals = sorted(set(labels)); mp = {v: i for i, v in enumerate(vals)}
        y = torch.tensor([mp[v] for v in labels])
        n_out = len(vals)
        counts = np.bincount(y.numpy())
        log(f"classes={vals} counts={counts.tolist()}")
        if counts.min() < args.folds:
            result["error"] = f"min class count {int(counts.min())} < folds {args.folds}"
            json.dump(result, open(args.out, "w"), indent=2); log("ABORT: " + result["error"]); return
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        split_iter = list(splitter.split(np.zeros(len(y)), y.numpy()))
    else:
        y = torch.tensor([float(v) for v in labels])
        n_out = 1
        splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
        split_iter = list(splitter.split(np.zeros(len(y))))

    all_pred = torch.zeros_like(y) if args.task == "classification" else torch.zeros(len(y))
    per_fold = []
    t_start = time.time()
    for i, (tr_idx, te_idx) in enumerate(split_iter):
        log(f"fold {i+1}/{len(split_iter)}: train={len(tr_idx)} test={len(te_idx)}")
        pred = run_fold(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx], args.task, dim, n_out,
                         args, dev, args.seed + i, f"{i+1}/{len(split_iter)}")
        all_pred[te_idx] = pred
        if args.task == "classification":
            fm = {"balanced_accuracy": float(balanced_accuracy_score(y[te_idx], pred)),
                  "macro_f1": float(f1_score(y[te_idx], pred, average="macro"))}
        else:
            yv, pv = y[te_idx].clamp_min(1e-6), pred.clamp_min(1e-6)
            m = float(mean_absolute_percentage_error(yv, pv))
            fm = {"mape": m, "score": max(0.0, 1 - m)}
        per_fold.append(fm)
        log(f"fold {i+1} done: {fm} (total elapsed {time.time()-t_start:.1f}s)")

    if args.task == "classification":
        result["pooled"] = {"balanced_accuracy": float(balanced_accuracy_score(y, all_pred)),
                             "macro_f1": float(f1_score(y, all_pred, average="macro")),
                             "class_counts": {str(v): int(c) for v, c in zip(vals, counts)}}
    else:
        yv, pv = y.clamp_min(1e-6), all_pred.clamp_min(1e-6)
        m = float(mean_absolute_percentage_error(yv, pv))
        result["pooled"] = {"mape": m, "score": max(0.0, 1 - m)}
    result["per_fold"] = per_fold
    result["total_seconds"] = time.time() - t_start
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    log("POOLED RESULT: " + json.dumps(result["pooled"]))


if __name__ == "__main__":
    main()
