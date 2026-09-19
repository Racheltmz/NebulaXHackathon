"""3-fold stratified CV for PS3 subsystems on frozen TimesFM3 representations.

Trains a FlaMinGo-style linear head (or other heads.py variants) directly on
top of cached TimesFM3 transformer features, with masking + noise
augmentation applied to the pooled representation during training for
few-shot robustness. Multi-channel inputs (e.g. Rail's 5 channels) are
handled natively since the head already pools over the channel axis with
learned per-channel embeddings.

Reports balanced accuracy + macro-F1 for classification (door/rail/acv) and
the official max(0, 1 - MAPE) score for regression (shm), both per-fold and
pooled across all held-out predictions.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_percentage_error
from heads import make_head


def augment(z, mask_p, noise_std):
    # z: [B, C, L, D] pooled feature tokens. Randomly mask whole channel/patch
    # tokens (structural dropout) and add Gaussian noise to the rest, so the
    # head cannot rely on any single token and generalizes from few examples.
    if mask_p > 0:
        keep = (torch.rand(z.shape[:3], device=z.device) > mask_p).float().unsqueeze(-1)
        z = z * keep / max(1e-3, 1 - mask_p)
    if noise_std > 0:
        z = z + noise_std * torch.randn_like(z)
    return z


def train_eval_fold(xtr, ytr, xte, yte, task, dim, channels, variant, qpc, epochs, mask_p, noise_std, dev, seed):
    torch.manual_seed(seed)
    if task == "classification":
        k = int(ytr.max().item()) + 1
        head = make_head(variant, dim, qpc).to(dev)
        q = nn.Parameter(torch.randn(k * qpc, head.query_dim, device=dev) * 0.02)
        counts = torch.bincount(ytr, minlength=k).float().clamp_min(1)
        loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(dev))
    else:
        head = make_head("linear", dim, qpc).to(dev)  # regression reuses linear head, 1 output query group
        q = nn.Parameter(torch.randn(qpc, head.query_dim, device=dev) * 0.02)
        loss_fn = nn.SmoothL1Loss()
        y_log = torch.log(ytr.clamp_min(1e-5))
        y_mean, y_std = y_log.mean(), y_log.std().clamp_min(1e-4)
        ytr_s = (y_log - y_mean) / y_std
    ce = nn.Parameter(torch.randn(channels, dim, device=dev) * 0.02)
    params = list(head.parameters()) + [ce, q]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    xtr, xte = xtr.to(dev), xte.to(dev)
    ytr_dev = (ytr if task == "classification" else ytr_s).to(dev)
    n = xtr.shape[0]
    bs = min(16, max(2, n))
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = augment(xtr[idx], mask_p, noise_std)
            pred = head(xb, ce, q)
            if task == "regression":
                pred = pred.reshape(-1)
            loss = loss_fn(pred, ytr_dev[idx].long() if task == "classification" else ytr_dev[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(xte, ce, q)
        if task == "regression":
            pred = torch.exp(pred.reshape(-1) * y_std + y_mean).cpu()
        else:
            pred = pred.argmax(-1).cpu()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--variant", default="flamingo_exact")
    ap.add_argument("--qpc", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--mask-p", type=float, default=0.2)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = Path(a.features)
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    # Fold on train+test pooled together: PS3's official held-out labels are
    # never released, so all locally-labeled examples become one pool for an
    # honest stratified CV estimate rather than a single lucky/unlucky split.
    feats = train["features"] + test["features"]
    labels = list(train["labels"]) + list(test["labels"])
    x = torch.stack([z.float() for z in feats])
    dim, channels = x.shape[-1], x.shape[1]

    result = {"task": fs.name, "variant": a.variant, "n": len(labels), "channels": channels,
              "mask_p": a.mask_p, "noise_std": a.noise_std, "folds": a.folds}

    if a.task == "classification":
        vals = sorted(set(labels)); mp = {v: i for i, v in enumerate(vals)}
        y = torch.tensor([mp[v] for v in labels])
        counts = np.bincount(y.numpy())
        if counts.min() < a.folds:
            result["error"] = f"min class count {counts.min()} < folds {a.folds}"
            json.dump(result, open(a.out, "w"), indent=2); print(json.dumps(result)); return
        splitter = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
        split_iter = splitter.split(np.zeros(len(y)), y.numpy())
    else:
        y = torch.tensor([float(v) for v in labels])
        splitter = KFold(a.folds, shuffle=True, random_state=a.seed)
        split_iter = splitter.split(np.zeros(len(y)))

    all_pred = torch.zeros_like(y) if a.task == "classification" else torch.zeros(len(y))
    per_fold = []
    for i, (tr_idx, te_idx) in enumerate(split_iter):
        pred = train_eval_fold(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx], a.task, dim, channels,
                                a.variant, a.qpc, a.epochs, a.mask_p, a.noise_std, dev, a.seed + i)
        all_pred[te_idx] = pred
        if a.task == "classification":
            per_fold.append({
                "balanced_accuracy": float(balanced_accuracy_score(y[te_idx], pred)),
                "macro_f1": float(f1_score(y[te_idx], pred, average="macro")),
            })
        else:
            yv = y[te_idx].clamp_min(1e-6); pv = pred.clamp_min(1e-6)
            m = float(mean_absolute_percentage_error(yv, pv))
            per_fold.append({"mape": m, "score": max(0.0, 1 - m)})

    if a.task == "classification":
        result["pooled"] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, all_pred)),
            "macro_f1": float(f1_score(y, all_pred, average="macro")),
            "class_counts": {str(v): int(c) for v, c in zip(vals, counts)},
        }
    else:
        yv = y.clamp_min(1e-6); pv = all_pred.clamp_min(1e-6)
        m = float(mean_absolute_percentage_error(yv, pv))
        result["pooled"] = {"mape": m, "score": max(0.0, 1 - m)}
    result["per_fold"] = per_fold
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
