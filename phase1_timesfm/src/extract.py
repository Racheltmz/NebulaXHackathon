"""Cache frozen TimesFM3 transformer features for UCR/UEA-style datasets."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

def load_model(timesfm_root, checkpoint, device):
    sys.path.insert(0, str(Path(timesfm_root) / "src"))
    from timesfm3.torch.model import TimesFM3Torch
    model = TimesFM3Torch.from_pretrained(checkpoint).to(device).eval()
    if not getattr(model, "use_variate_attention", False):
        raise RuntimeError("TimesFM3 checkpoint is not using variate attention; refusing multivariate feature extraction")
    for p in model.parameters(): p.requires_grad_(False)
    return model

def features(model, x, device, context=512, normalize=True):
    # x is [C,T]. One fixed context makes cached heads simple and reproducible.
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    if normalize: x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-5)
    x = x[:, -context:]
    if x.shape[1] < context:
        x = np.pad(x, ((0, 0), (context - x.shape[1], 0)))
    p = model.input_patch_len
    n = context // p
    values = torch.from_numpy(x).to(device).reshape(1, x.shape[0], n, p)
    masks = torch.zeros_like(values, dtype=torch.bool)
    # Every observed channel is a target series.  This keeps the forward pass
    # on TimesFM3's intended multivariate target path; patch_is_target=False
    # denotes covariates and changes the rolled/future masking behavior.
    target = torch.ones((1, x.shape[0], n), device=device, dtype=torch.bool)
    with torch.inference_mode():
        out = model({"values": values, "masks": masks, "patch_is_target": target},
                    return_aux_outputs=True)
    return out["__call__:transformer_output"][0].cpu().half()

def features_batch(model, xs, device, context=512, normalize=True, max_stored_channels=8):
    """Batched equivalent of features(); all UCR examples are fixed-width."""
    arr = []
    for x in xs:
        x = np.nan_to_num(np.asarray(x, dtype=np.float32))[:, -context:]
        if normalize: x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-5)
        if x.shape[1] < context:
            x = np.pad(x, ((0, 0), (context - x.shape[1], 0)))
        arr.append(x)
    x = np.stack(arr)
    p = model.input_patch_len; n = context // p
    values = torch.from_numpy(x).to(device).reshape(len(arr), x.shape[1], n, p)
    masks = torch.zeros_like(values, dtype=torch.bool)
    target = torch.ones((len(arr), x.shape[1], n), device=device, dtype=torch.bool)
    with torch.inference_mode():
        out = model({"values": values, "masks": masks, "patch_is_target": target},
                    return_aux_outputs=True)
        z = out["__call__:transformer_output"]
        # TimesFM3 still sees every original channel above.  For very
        # high-channel UEA problems, retaining one 1024-d vector for every
        # channel/example is infeasible (e.g. FaceDetection is >100 GB).
        # Store four robust channel-distribution tokens after the multivariate
        # backbone: mean, std, max, and min. Low-channel datasets are kept
        # exactly as produced by the backbone.
        if z.shape[1] > max_stored_channels:
            z = torch.stack((z.mean(1), z.std(1, unbiased=False), z.amax(1), z.amin(1)), dim=1)
        return z.cpu().half()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--timesfm-root", required=True); ap.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    ap.add_argument("--context", type=int, default=512); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--device", default="cuda"); ap.add_argument("--no-normalize", action="store_true"); ap.add_argument("--max-stored-channels", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.output, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = load_model(a.timesfm_root, a.checkpoint, device)
    meta = json.load(open(a.input))
    for name, ds in meta.items():
        out = Path(a.output) / name; out.mkdir(exist_ok=True)
        channels = int(np.asarray(np.load(ds["train_x"], allow_pickle=True)[0]).shape[0])
        # Bound the largest [batch, channel] product. UEA includes 144-,
        # 200-, and 963-channel problems; a fixed batch of 32 can exceed GPU
        # memory even though ordinary UCR/UEA datasets fit comfortably.
        dataset_batch = max(1, min(a.batch_size, 512 // max(1, channels)))
        print(name, "channels", channels, "batch", dataset_batch, flush=True)
        for split in ("train", "test"):
            x = np.load(ds[f"{split}_x"], allow_pickle=True)
            y = np.load(ds[f"{split}_y"], allow_pickle=True)
            fs = []
            for start in range(0, len(x), dataset_batch):
                fs.extend(features_batch(model, x[start:start + dataset_batch], device, a.context, not a.no_normalize, a.max_stored_channels))
            torch.save({"features": fs, "labels": y.tolist(), "channels": int(x[0].shape[0])}, out / f"{split}.pt")
        print(name, len(x), "->", out, flush=True)

if __name__ == "__main__": main()
