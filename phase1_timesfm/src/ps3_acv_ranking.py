"""ACV evaluated the way it's actually scored: per-case ranking, leave-one-case-out.

Prior CV bug: ACV has exactly 6 cases x 8 cars = 48 examples, and every car's
feature is a residual computed relative to the OTHER 7 cars in its own case
((car - median_case)/MAD_case, see ps3_adapter.py). A plain StratifiedKFold
on the pooled 48 examples can and did split cars from the same case across
train/test -- leaking within-case statistics, and evaluating something other
than the real task (which only ever sees a whole unseen case at inference).

Fix: leave-one-case-out CV (6 folds, one per case) -- this is dictated by the
data (there are only 6 groups), not a free hyperparameter. For each held-out
case, train on the other 5 cases' 40 cars, then rank that case's 8 cars by
predicted P(faulty) and score with the official formula:
    score = (n - (r - 1)) / n   where r = rank of the true faulty car (1=top)
Also reports the classification-style BAcc/macro-F1 for continuity with the
rest of the sweep, but the ranking score is the number that matters here.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ps3_dl import Model, mask_patches, mixup, log


def run_fold(xtr, ytr, xte, task, dim, n_out, args, dev, seed, class_weight):
    torch.manual_seed(seed)
    model = Model(args.arch, dim, n_out, hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xtr, xte = xtr.to(dev), xte.to(dev)

    with torch.no_grad():
        pooled_tr_raw = model.pool(xtr)
        mean, std = pooled_tr_raw.mean(0, keepdim=True), pooled_tr_raw.std(0, keepdim=True).clamp_min(1e-4)

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
            pooled = (model.pool(xb) - mean) / std
            pooled, target, sw = mixup(pooled, y_onehot[idx], sample_w_all[idx], args.mixup_alpha)
            logits = model(pooled)
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
        pooled_te = (model.pool(xte) - mean) / std
        probs = torch.softmax(model(pooled_te), -1)[:, 1].cpu()  # P(faulty)
    return probs


def rank_decay_score(probs, y_true):
    n = len(probs)
    order = np.argsort(-probs)  # descending by P(faulty)
    faulty_idx = int(np.where(y_true == 1)[0][0])
    r = int(np.where(order == faulty_idx)[0][0]) + 1  # 1-indexed rank
    return (n - (r - 1)) / n, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--arch", default="attn")
    ap.add_argument("--cars-per-case", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
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
    dim = x.shape[-1]

    n_total = len(y)
    cpc = args.cars_per_case
    assert n_total % cpc == 0, f"n={n_total} not divisible by cars-per-case={cpc}"
    n_cases = n_total // cpc
    groups = np.repeat(np.arange(n_cases), cpc)
    log(f"n={n_total} cases={n_cases} cars_per_case={cpc} arch={args.arch}")

    scores, per_case = [], []
    t0 = time.time()
    for case_i in range(n_cases):
        te_idx = np.where(groups == case_i)[0]
        tr_idx = np.where(groups != case_i)[0]
        ytr = y[tr_idx]
        counts = torch.bincount(ytr, minlength=len(vals)).float().clamp_min(1)
        class_weight = counts.sum() / (len(vals) * counts)
        probs = run_fold(x[tr_idx], ytr, x[te_idx], "classification", dim, len(vals),
                          args, dev, args.seed + case_i, class_weight)
        y_case = y[te_idx].numpy()
        score, r = rank_decay_score(probs.numpy(), y_case)
        pred_hard = (probs > 0.5).long().numpy()
        bacc = float(balanced_accuracy_score(y_case, pred_hard)) if len(set(y_case)) > 1 else None
        scores.append(score)
        per_case.append({"case": case_i, "rank_decay_score": score, "true_faulty_rank": r,
                          "n_cars": len(te_idx), "balanced_accuracy": bacc})
        log(f"case {case_i}: rank={r}/{len(te_idx)} score={score:.4f} (elapsed {time.time()-t0:.1f}s)")

    result = {"task": "acv", "arch": args.arch, "n_cases": n_cases,
              "mean_rank_decay_score": float(np.mean(scores)),
              "per_case": per_case}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    log(f"MEAN RANK-DECAY SCORE [{args.arch}]: {result['mean_rank_decay_score']:.4f}")


if __name__ == "__main__":
    main()
