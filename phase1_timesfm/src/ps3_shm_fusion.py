"""SHM regression with raw-signal statistics fused into the pooled embedding.

EDA finding motivating this: raw peak amplitude correlates r=0.871 with the
cumulative-damage target (RMS/std/mean_abs all ~0.7), yet the frozen
TimesFM3 embedding alone only gets a 0.300 score. Rather than switching to a
classical model on those stats (ruled out), fuse them as four extra input
features into the same deep head: concat([pooled_embedding, raw_stats]) ->
MLP -> scalar. Still an end-to-end trained neural network; the stats are
just additional evidence for it, not a competing model.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ps3_dl import ChannelAttnPool, MILAttnPool, mask_patches, mixup, log


def raw_stats(x_list):
    rows = []
    for ex in x_list:
        ex = np.nan_to_num(np.asarray(ex, dtype=np.float64))
        flat = ex.reshape(-1)
        rms = np.sqrt((flat ** 2).mean())
        peak = np.abs(flat).max()
        std = flat.std()
        mean_abs = np.abs(flat).mean()
        rows.append([np.log1p(rms), np.log1p(peak), np.log1p(std), np.log1p(mean_abs)])
    return np.array(rows, dtype=np.float32)


class FusionModel(nn.Module):
    def __init__(self, dim, n_stats, backbone="mil", hidden=128, dropout=0.2):
        super().__init__()
        self.pool = MILAttnPool(dim) if backbone == "mil" else ChannelAttnPool(dim)
        feat_dim = 2 * dim + n_stats
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, tokens, stats):
        pooled = self.pool(tokens)
        return self.head(torch.cat([pooled, stats], -1)), pooled


def run_fold(xtr, str_, ytr, xte, ste_, args, dev, seed):
    torch.manual_seed(seed)
    dim = xtr.shape[-1]
    model = FusionModel(dim, str_.shape[1], args.backbone, args.hidden, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xtr, xte = xtr.to(dev), xte.to(dev)

    with torch.no_grad():
        pooled_raw = model.pool(xtr)
        p_mean, p_std = pooled_raw.mean(0, keepdim=True), pooled_raw.std(0, keepdim=True).clamp_min(1e-4)
    s_mean, s_std = str_.mean(0, keepdim=True), str_.std(0, keepdim=True).clamp_min(1e-4)
    str_n = ((str_ - s_mean) / s_std).to(dev)
    ste_n = ((ste_ - s_mean) / s_std).to(dev)

    y_log = torch.log(ytr.clamp_min(1e-5))
    y_mean, y_std = y_log.mean(), y_log.std().clamp_min(1e-4)
    ytr_dev = ((y_log - y_mean) / y_std).to(dev).unsqueeze(-1)

    n = xtr.shape[0]
    bs = min(args.batch_size, max(2, n))
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = mask_patches(xtr[idx], args.mask_p)
            xb = xb + args.noise_std * torch.randn_like(xb)
            pred, pooled = model(xb, str_n[idx])
            pooled_n = (pooled - p_mean) / p_std
            loss = nn.functional.smooth_l1_loss(pred, ytr_dev[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred, _ = model(xte, ste_n)
        pred = torch.exp(pred.squeeze(-1) * y_std + y_mean).cpu()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/ps3_features/shm")
    ap.add_argument("--arrays-root", default="data/ps3_arrays")
    ap.add_argument("--backbone", choices=["linear", "mil"], default="mil")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mask-p", type=float, default=0.1)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = Path(args.features)
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    x = torch.stack([z.float() for z in (train["features"] + test["features"])])
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    y = torch.tensor([float(v) for v in (list(train["labels"]) + list(test["labels"]))])

    raw_tr = np.load(Path(args.arrays_root) / "shm_train_x.npy", allow_pickle=True)
    raw_te = np.load(Path(args.arrays_root) / "shm_test_x.npy", allow_pickle=True)
    stats = torch.from_numpy(raw_stats(list(raw_tr) + list(raw_te)))
    log(f"n={len(y)} stats_dim={stats.shape[1]}")

    splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
    all_pred = torch.zeros(len(y))
    per_fold = []
    t0 = time.time()
    for i, (tr_idx, te_idx) in enumerate(splitter.split(np.zeros(len(y)))):
        pred = run_fold(x[tr_idx], stats[tr_idx], y[tr_idx], x[te_idx], stats[te_idx], args, dev, args.seed + i)
        all_pred[te_idx] = pred
        yv, pv = y[te_idx].clamp_min(1e-6), pred.clamp_min(1e-6)
        m = float(mean_absolute_percentage_error(yv, pv))
        per_fold.append({"mape": m, "score": max(0.0, 1 - m)})
        log(f"fold {i+1} done: {per_fold[-1]} (elapsed {time.time()-t0:.1f}s)")

    yv, pv = y.clamp_min(1e-6), all_pred.clamp_min(1e-6)
    m = float(mean_absolute_percentage_error(yv, pv))
    result = {"task": "shm", "backbone": args.backbone, "n": len(y),
              "pooled": {"mape": m, "score": max(0.0, 1 - m)}, "per_fold": per_fold}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    log("POOLED RESULT (fusion): " + json.dumps(result["pooled"]))


if __name__ == "__main__":
    main()
