"""Cache frozen TimesFM3 features in TRUE univariate mode for comparison.

The existing extract.py feeds all of an example's channels to the model at
once with use_variate_attention=True, so the backbone cross-mixes channels
before we ever see the embedding (confirmed via ps3_channel_corr.py: raw
channels are near-independent but the resulting embeddings are highly
correlated, 0.66-0.88 cosine sim). This script instead runs each channel
through the model on its own (channel-count=1 per forward pass), so variate
attention has nothing to mix with and the output is a genuine per-channel
representation. Output shape/format matches extract.py's cache exactly
([examples] of [C, n_patches, dim] tensors) so it's a drop-in swap for
ps3_dl.py -- only the extraction differs, not the downstream pooling/head.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch


def load_model(timesfm_root, checkpoint, device):
    sys.path.insert(0, str(Path(timesfm_root) / "src"))
    from timesfm3.torch.model import TimesFM3Torch
    model = TimesFM3Torch.from_pretrained(checkpoint).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def normalize_pad(x, context, normalize):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    if normalize:
        x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-5)
    x = x[:, -context:]
    if x.shape[1] < context:
        x = np.pad(x, ((0, 0), (context - x.shape[1], 0)))
    return x


def extract_univariate(model, xs, device, context=512, normalize=True, batch_size=64):
    """xs: list of [C,T] arrays. Returns list of [C, n_patches, dim] tensors,
    each channel passed through the model independently (channel dim = 1)."""
    p = model.input_patch_len
    n = context // p
    # flatten (example, channel) into one batch axis of single-channel series
    flat_channels, owners = [], []  # owners[i] = example index the i-th channel row belongs to
    per_example_channels = []
    for ei, x in enumerate(xs):
        x = normalize_pad(x, context, normalize)
        per_example_channels.append(x.shape[0])
        for c in range(x.shape[0]):
            flat_channels.append(x[c:c + 1])  # [1, context]
            owners.append(ei)
    out_rows = [None] * len(flat_channels)
    for start in range(0, len(flat_channels), batch_size):
        chunk = flat_channels[start:start + batch_size]
        arr = np.stack(chunk)  # [b, 1, context]
        values = torch.from_numpy(arr).to(device).reshape(len(chunk), 1, n, p)
        masks = torch.zeros_like(values, dtype=torch.bool)
        target = torch.ones((len(chunk), 1, n), device=device, dtype=torch.bool)
        with torch.inference_mode():
            out = model({"values": values, "masks": masks, "patch_is_target": target},
                        return_aux_outputs=True)
            z = out["__call__:transformer_output"]  # [b, 1, n, dim]
        for i in range(len(chunk)):
            out_rows[start + i] = z[i, 0].cpu().half()  # [n, dim]
    # regroup rows back into per-example [C, n_patches, dim]
    results, idx = [], 0
    for c in per_example_channels:
        results.append(torch.stack(out_rows[idx:idx + c]))
        idx += c
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arrays-root", default="data/ps3_arrays")
    ap.add_argument("--output-root", default="data/ps3_features_univariate")
    ap.add_argument("--timesfm-root", default="/home/s/saad304/timesfm")
    ap.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"loading model on {device}...", flush=True)
    model = load_model(a.timesfm_root, a.checkpoint, device)

    out_dir = Path(a.output_root) / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        x = np.load(Path(a.arrays_root) / f"{a.task}_{split}_x.npy", allow_pickle=True)
        y = np.load(Path(a.arrays_root) / f"{a.task}_{split}_y.npy", allow_pickle=True)
        print(f"{a.task} {split}: {len(x)} examples", flush=True)
        feats = extract_univariate(model, x, device, a.context, True, a.batch_size)
        torch.save({"features": feats, "labels": y.tolist()}, out_dir / f"{split}.pt")
        print(f"  -> saved {out_dir / (split + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
