"""Adapt a trained frozen-TimesFM head on a deferred raw UEA dataset.

This script is deliberately for datasets without a complete feature cache.
Completed UCR/UEA embeddings remain consumed by the cached training path.
Raw batches are converted in TimesFM3 multivariate mode and immediately
released, so feature tensors are never accumulated for the whole dataset.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from heads import make_head
from extract import load_model, features_batch

def stable_map(train_y, test_y):
    vals = sorted(set(train_y.tolist()) | set(test_y.tolist()))
    return {v: i for i, v in enumerate(vals)}

def raw_batch(meta, name, split, indices, label_map, model, device, context, batch_size):
    x = np.load(meta[name][f"{split}_x"], allow_pickle=True)
    y = np.load(meta[name][f"{split}_y"], allow_pickle=True)
    out_x, out_y = [], []
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        z = features_batch(model, [x[int(i)] for i in ix], device,
                           context=context, normalize=True,
                           max_stored_channels=8).float()
        out_x.append(z)
        out_y.extend(label_map[y[int(i)]] for i in ix)
    return torch.cat(out_x), torch.tensor(out_y, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--raw-manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--timesfm-root", required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(a.raw_manifest)); name = a.dataset
    train_y = np.load(meta[name]["train_y"], allow_pickle=True)
    test_y = np.load(meta[name]["test_y"], allow_pickle=True)
    label_map = stable_map(train_y, test_y); k = len(label_map)
    model = load_model(a.timesfm_root, "google/timesfm-3.0-pytorch", device)
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    variant, dim, qpc = ckpt["variant"], ckpt["dim"], ckpt["qpc"]
    head = make_head(variant, dim, qpc).to(device)
    head.load_state_dict(ckpt["head_state_dict"]); head.eval()
    tx = np.load(meta[name]["train_x"], allow_pickle=True)
    probe = features_batch(model, [tx[0]], device, a.context, True, 8)
    channels = int(probe.shape[1])
    adapter = nn.Parameter(torch.zeros(channels, dim, device=device))
    queries = nn.Parameter(torch.randn(k * qpc, getattr(head, "query_dim", dim), device=device) * 0.02)
    opt = torch.optim.AdamW([adapter, queries], lr=a.lr)
    tr_ix = np.arange(len(train_y))
    for _ in range(a.adapt_steps):
        take = np.random.choice(tr_ix, size=min(64, len(tr_ix)), replace=False)
        xb, yb = raw_batch(meta, name, "train", take, label_map, model, device, a.context, a.batch_size)
        # The head adds the dataset-local channel adapter internally.  Do not
        # add it to the backbone features a second time.
        logits = head(xb.to(device), adapter, queries)
        if a.balanced:
            counts = torch.bincount(yb, minlength=k).float().clamp_min(1).to(device)
            weight = (len(yb) / (k * counts)).to(device)
            loss = nn.functional.cross_entropy(logits, yb.to(device), weight=weight)
        else:
            loss = nn.functional.cross_entropy(logits, yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    te_ix = np.arange(len(test_y))
    xb, yb = raw_batch(meta, name, "test", te_ix, label_map, model, device, a.context, a.batch_size)
    with torch.no_grad():
        pred = head(xb.to(device), adapter, queries).argmax(-1).cpu().numpy()
    result = {"dataset": name, "variant": variant, "metrics": {
        "accuracy": accuracy_score(yb.numpy(), pred),
        "balanced_accuracy": balanced_accuracy_score(yb.numpy(), pred),
        "macro_f1": f1_score(yb.numpy(), pred, average="macro")},
        "mode": "on_the_fly_timesfm3_multivariate", "channels_stored": channels,
        "adapt_steps": a.adapt_steps}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2); print(json.dumps(result), flush=True)

if __name__ == "__main__": main()
