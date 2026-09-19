import json
from pathlib import Path
import numpy as np
import torch

train = torch.load("data/ps3_features/rail/train.pt", weights_only=False)
test = torch.load("data/ps3_features/rail/test.pt", weights_only=False)
feats = train["features"] + test["features"]
labels = list(train["labels"]) + list(test["labels"])
x = torch.stack([z.float() for z in feats])
x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
chan_vec = x.mean(2).norm(dim=-1).numpy()

vals = sorted(set(labels))
y = np.array([vals.index(v) for v in labels])
maxd = np.zeros(chan_vec.shape[1])
for v in range(len(vals)):
    g0, g1 = chan_vec[y == v], chan_vec[y != v]
    pooled_std = np.sqrt((g0.var(0) + g1.var(0)) / 2) + 1e-6
    d = np.abs(g0.mean(0) - g1.mean(0)) / pooled_std
    maxd = np.maximum(maxd, d)

order = np.argsort(-maxd)
print("full ranking (channel, |d|):")
for c in order:
    print(f"  {c}: {maxd[c]:.4f}")
json.dump({"order": order.tolist(), "d": maxd.tolist()}, open("reports/rail-channel-rank.json", "w"))
