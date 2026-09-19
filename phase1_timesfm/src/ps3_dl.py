"""Deep-learning classifier/regressor on frozen TimesFM3 features for PS3.

Iterates architecture in increasing complexity, all sharing the same pooling
contract so they're a fair comparison under identical 3-fold stratified CV:

  --arch linear : channel-attention pool -> single Linear (FlaMinGo-style probe)
  --arch mlp    : channel-attention pool -> Linear -> GELU -> Dropout -> Linear
  --arch attn   : one self-attention layer lets per-channel tokens attend to
                  each other before the same channel-attention pool -> MLP head
                  (this is what lets Rail's 5 channels / ACV's cross-car
                  channels share information instead of being pooled blindly)
  --arch cnn    : a small 1D conv over the patch axis per channel extracts
                  local temporal patterns before pooling -> MLP head
  --arch mil    : gated-attention MIL pooling (Ilse et al. 2018) over
                  channels-as-instances -- each channel gets a learned
                  relevance gate tanh(Vh)*sigmoid(Uh) instead of a plain
                  linear score. This is the right tool specifically when
                  channels are closer to independent/exchangeable "bag
                  members" than to a sequence that needs pairwise
                  cross-channel attention (see ps3_channel_corr.py for the
                  empirical check that motivates choosing this over --arch attn)

Class-imbalance handling: classification loss is inverse-frequency weighted
(critical for Rail 234/14/24 and ACV 42/6 -- unweighted CE collapses to the
majority class, which is exactly what happened before this fix).

Low-data robustness: structured (channel, patch) token masking, Gaussian
noise on pooled features, mixup (with sample reweighting so it composes
correctly with class weighting), label smoothing, dropout, weight decay.

Evaluation: 3-fold stratified CV (KFold for regression), pooled + per-fold
balanced accuracy / macro-F1 (classification) or the official
max(0, 1-MAPE) score (regression). Per-epoch logging with ETA.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_percentage_error


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ChannelAttnPool(nn.Module):
    """Mean+max pool over patches per channel, then softmax-weighted pool over channels."""
    def __init__(self, dim):
        super().__init__()
        self.channel_score = nn.Linear(dim, 1)

    def forward(self, tokens):
        # tokens: [B, C, L, D]
        mean_p = tokens.mean(2)
        max_p = tokens.amax(2)
        pooled_ch = torch.cat([mean_p, max_p], -1)          # [B, C, 2D]
        weights = torch.softmax(self.channel_score(mean_p), dim=1)  # [B, C, 1]
        return (pooled_ch * weights).sum(1)                  # [B, 2D]


class AttnBackbone(nn.Module):
    """One self-attention layer over per-channel tokens, then ChannelAttnPool."""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.channel_token = nn.Sequential(nn.LayerNorm(dim))
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.pool = ChannelAttnPool(dim)

    def forward(self, tokens):
        b, c, l, d = tokens.shape
        chan_tok = self.channel_token(tokens.mean(2))         # [B, C, D]
        attended, _ = self.attn(chan_tok, chan_tok, chan_tok)
        chan_tok = self.norm(chan_tok + attended)              # [B, C, D]
        # feed the attended channel summary back as a single "patch" per
        # channel so ChannelAttnPool's mean/max pooling still applies
        return self.pool(chan_tok.unsqueeze(2).expand(-1, -1, l, -1))


class MILAttnPool(nn.Module):
    """Gated-attention MIL pooling over channels-as-instances (Ilse et al. 2018).

    gate = tanh(V h) * sigmoid(U h); score = w^T gate; weights = softmax(score).
    The sigmoid term lets a channel be zeroed out entirely (a hard gate) while
    tanh keeps the scoring signed -- strictly more expressive than the plain
    linear channel_score in ChannelAttnPool, at only a small parameter cost,
    which fits channels that are semantically instances of a bag (this task's
    channels) rather than a sequence that needs full pairwise self-attention.
    """
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, tokens):
        mean_p = tokens.mean(2)
        max_p = tokens.amax(2)
        pooled_ch = torch.cat([mean_p, max_p], -1)
        gate = torch.tanh(self.V(mean_p)) * torch.sigmoid(self.U(mean_p))
        weights = torch.softmax(self.w(gate), dim=1)          # [B, C, 1]
        return (pooled_ch * weights).sum(1)


class CNNBackbone(nn.Module):
    """1D conv over the patch axis (per channel) before ChannelAttnPool."""
    def __init__(self, dim, hidden=None, kernel=3, dropout=0.1):
        super().__init__()
        hidden = hidden or dim
        self.conv = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel, padding=kernel // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, kernel, padding=kernel // 2), nn.GELU(),
        )
        self.pool = ChannelAttnPool(dim)

    def forward(self, tokens):
        b, c, l, d = tokens.shape
        flat = tokens.reshape(b * c, l, d).transpose(1, 2)     # [B*C, D, L]
        conv_out = self.conv(flat).transpose(1, 2).reshape(b, c, l, d)
        return self.pool(conv_out)


class Model(nn.Module):
    def __init__(self, arch, dim, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.arch = arch
        if arch == "linear":
            self.backbone = ChannelAttnPool(dim)
            feat_dim = 2 * dim
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, n_out))
        elif arch == "mlp":
            self.backbone = ChannelAttnPool(dim)
            feat_dim = 2 * dim
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, n_out))
        elif arch == "attn":
            self.backbone = AttnBackbone(dim, dropout=dropout)
            feat_dim = 2 * dim
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, n_out))
        elif arch == "cnn":
            self.backbone = CNNBackbone(dim, dropout=dropout)
            feat_dim = 2 * dim
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, n_out))
        elif arch == "mil":
            self.backbone = MILAttnPool(dim)
            feat_dim = 2 * dim
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, n_out))
        else:
            raise ValueError(arch)
        self.feat_dim = feat_dim

    def pool(self, tokens):
        return self.backbone(tokens)

    def forward(self, pooled_norm):
        return self.head(pooled_norm)


def mask_patches(tokens, p):
    if p <= 0:
        return tokens
    keep = (torch.rand(tokens.shape[:3], device=tokens.device) > p).float().unsqueeze(-1)
    return tokens * keep / max(1e-3, 1 - p)


def mixup(x, y_onehot, sample_w, alpha):
    if alpha <= 0:
        return x, y_onehot, sample_w
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    return (lam * x + (1 - lam) * x[perm],
            lam * y_onehot + (1 - lam) * y_onehot[perm],
            lam * sample_w + (1 - lam) * sample_w[perm])


def run_fold(xtr, ytr, xte, yte, task, dim, n_out, args, dev, seed, fold_tag, class_weight=None):
    torch.manual_seed(seed)
    model = Model(args.arch, dim, n_out, hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xtr, xte = xtr.to(dev), xte.to(dev)

    with torch.no_grad():
        pooled_tr_raw = model.pool(xtr)
        mean, std = pooled_tr_raw.mean(0, keepdim=True), pooled_tr_raw.std(0, keepdim=True).clamp_min(1e-4)

    if task == "classification":
        ytr_dev, yte_dev = ytr.to(dev), yte.to(dev)
        y_onehot = torch.nn.functional.one_hot(ytr_dev, n_out).float()
        cw = class_weight.to(dev) if class_weight is not None else torch.ones(n_out, device=dev)
        sample_w_all = cw[ytr_dev]
    else:
        y_log = torch.log(ytr.clamp_min(1e-5))
        y_mean, y_std = y_log.mean(), y_log.std().clamp_min(1e-4)
        ytr_dev = ((y_log - y_mean) / y_std).to(dev).unsqueeze(-1)
        yte_dev = yte.to(dev)
        sample_w_all = torch.ones(len(ytr), device=dev)

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
            ep_loss += float(loss.detach()) / steps_per_epoch
        sched.step()
        if args.log_every and (ep + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ep + 1) * (args.epochs - ep - 1)
            log(f"    [{args.arch}] fold {fold_tag} epoch {ep+1}/{args.epochs} loss={ep_loss:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    model.eval()
    with torch.no_grad():
        pooled_te = (model.pool(xte) - mean) / std
        if task == "classification":
            pred = model(pooled_te).argmax(-1).cpu()
        else:
            pred = torch.exp(model(pooled_te).squeeze(-1) * y_std + y_mean).cpu()
    return pred


def load_xy(features_path, mode):
    fs = Path(features_path)
    train = torch.load(fs / "train.pt", weights_only=False)
    test = torch.load(fs / "test.pt", weights_only=False)
    feats = train["features"] + test["features"]
    labels = list(train["labels"]) + list(test["labels"])
    x = torch.stack([z.float() for z in feats])
    n_bad = int((~torch.isfinite(x)).sum())
    if n_bad:
        log(f"WARNING: {n_bad} non-finite feature values, clamping")
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
    if mode == "classification":
        vals = sorted(set(labels)); mp = {v: i for i, v in enumerate(vals)}
        y = torch.tensor([mp[v] for v in labels])
        return x, y, vals
    return x, torch.tensor([float(v) for v in labels]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--arch", choices=["linear", "mlp", "attn", "cnn", "mil"], default="linear")
    ap.add_argument("--folds", type=int, default=3)
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
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} task={args.task} arch={args.arch} features={args.features}")

    x, y, vals = load_xy(args.features, args.task)
    dim = x.shape[-1]
    log(f"n={len(y)} channels={x.shape[1]} patches={x.shape[2]} dim={dim}")

    hp = dict(vars(args)); del hp["features"], hp["out"]
    result = {"task": Path(args.features).name, "n": len(y), "channels": int(x.shape[1]), "hyperparams": hp}

    if args.task == "classification":
        counts = np.bincount(y.numpy())
        log(f"classes={vals} counts={counts.tolist()}")
        if counts.min() < args.folds:
            result["error"] = f"min class count {int(counts.min())} < folds {args.folds}"
            json.dump(result, open(args.out, "w"), indent=2); log("ABORT: " + result["error"]); return
        n_out = len(vals)
        class_weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        log(f"class weights={class_weight.tolist()}")
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        split_iter = list(splitter.split(np.zeros(len(y)), y.numpy()))
    else:
        n_out, class_weight = 1, None
        splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
        split_iter = list(splitter.split(np.zeros(len(y))))

    all_pred = torch.zeros_like(y) if args.task == "classification" else torch.zeros(len(y))
    per_fold = []
    t_start = time.time()
    for i, (tr_idx, te_idx) in enumerate(split_iter):
        log(f"fold {i+1}/{len(split_iter)}: train={len(tr_idx)} test={len(te_idx)}")
        pred = run_fold(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx], args.task, dim, n_out,
                         args, dev, args.seed + i, f"{i+1}/{len(split_iter)}", class_weight)
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
    log(f"POOLED RESULT [{args.arch}]: " + json.dumps(result["pooled"]))


if __name__ == "__main__":
    main()
