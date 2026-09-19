"""Check channel independence for PS3 multi-channel tasks.

Two complementary checks, since either could justify a channel-independence
assumption on its own but they can disagree:
  1. Raw input correlation -- Pearson correlation between channels' time
     series, per example, averaged. High |corr| means channels carry
     redundant information at the signal level.
  2. Frozen-embedding correlation -- cosine similarity between channels'
     mean-pooled TimesFM3 embeddings. This is what the trained head actually
     consumes, so it's the more decision-relevant number: even if raw
     signals are independent, the backbone may have mixed information across
     channels (it does -- variate attention), making the embeddings
     correlated regardless.

Verdict heuristic: mean |off-diagonal correlation| < 0.2 -> channels look
close to independent (pooling/summing per-channel, no cross-channel term,
is a reasonable simplification); > 0.5 -> channels are highly redundant or
structurally coupled (a shared/attention pooling across channels is doing
real work, not just adding capacity); in between -> mixed, architecture
choice should be validated empirically per task (which is what the fast
probes already did).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch


def raw_channel_corr(x_path):
    arr = np.load(x_path, allow_pickle=True)
    per_example = []
    for ex in arr:
        ex = np.nan_to_num(np.asarray(ex, dtype=np.float64))
        if ex.shape[0] < 2 or ex.shape[1] < 3:
            continue
        c = np.corrcoef(ex)
        c = np.nan_to_num(c)
        n = c.shape[0]
        off_diag = c[~np.eye(n, dtype=bool)]
        per_example.append(np.abs(off_diag).mean())
    return per_example


def embedding_channel_corr(features_path):
    train = torch.load(Path(features_path) / "train.pt", weights_only=False)
    test = torch.load(Path(features_path) / "test.pt", weights_only=False)
    feats = train["features"] + test["features"]
    per_example = []
    for f in feats:
        f = torch.nan_to_num(f.float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        chan_vec = f.mean(1)  # [C, D] -- mean over patches
        if chan_vec.shape[0] < 2:
            continue
        norm = chan_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
        unit = chan_vec / norm
        sim = unit @ unit.T
        n = sim.shape[0]
        off_diag = sim[~torch.eye(n, dtype=torch.bool)]
        per_example.append(float(off_diag.abs().mean()))
    return per_example


def verdict(mean_abs):
    if mean_abs < 0.2:
        return "near-independent"
    if mean_abs > 0.5:
        return "highly coupled"
    return "mixed / moderate coupling"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arrays-root", default="data/ps3_arrays")
    ap.add_argument("--features-root", default="data/ps3_features")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result = {"task": a.task}

    raw_path = Path(a.arrays_root) / f"{a.task}_train_x.npy"
    if raw_path.exists():
        raw_corrs = raw_channel_corr(raw_path)
        raw_test = Path(a.arrays_root) / f"{a.task}_test_x.npy"
        if raw_test.exists():
            raw_corrs += raw_channel_corr(raw_test)
        if raw_corrs:
            m = float(np.mean(raw_corrs))
            result["raw_signal"] = {"mean_abs_offdiag_corr": m, "std": float(np.std(raw_corrs)),
                                     "n_examples": len(raw_corrs), "verdict": verdict(m)}

    emb_corrs = embedding_channel_corr(Path(a.features_root) / a.task)
    if emb_corrs:
        m = float(np.mean(emb_corrs))
        result["timesfm3_embedding"] = {"mean_abs_offdiag_cosine": m, "std": float(np.std(emb_corrs)),
                                         "n_examples": len(emb_corrs), "verdict": verdict(m)}

    print(json.dumps(result, indent=2), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
