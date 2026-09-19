"""Cache frozen TimesFM3 transformer features for UCR/UEA-style datasets."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

def load_model(timesfm_root, checkpoint, device):
    sys.path.insert(0, str(Path(timesfm_root) / "src"))
    from timesfm3.torch.model import TimesFM3Torch
    model = TimesFM3Torch.from_pretrained(checkpoint).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def features(model, x, device, context=512):
    # x is [C,T]. One fixed context makes cached heads simple and reproducible.
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    x = x[:, -context:]
    if x.shape[1] < context:
        x = np.pad(x, ((0, 0), (context - x.shape[1], 0)))
    p = model.input_patch_len
    n = context // p
    values = torch.from_numpy(x).to(device).reshape(1, x.shape[0], n, p)
    masks = torch.zeros_like(values, dtype=torch.bool)
    target = torch.zeros((1, x.shape[0], n), device=device, dtype=torch.bool)
    with torch.inference_mode():
        out = model({"values": values, "masks": masks, "patch_is_target": target},
                    return_aux_outputs=True)
    return out["__call__:transformer_output"][0].cpu().half()

def features_batch(model, xs, device, context=512):
    """Batched equivalent of features(); all UCR examples are fixed-width."""
    arr = []
    for x in xs:
        x = np.nan_to_num(np.asarray(x, dtype=np.float32))[:, -context:]
        if x.shape[1] < context:
            x = np.pad(x, ((0, 0), (context - x.shape[1], 0)))
        arr.append(x)
    x = np.stack(arr)
    p = model.input_patch_len; n = context // p
    values = torch.from_numpy(x).to(device).reshape(len(arr), x.shape[1], n, p)
    masks = torch.zeros_like(values, dtype=torch.bool)
    target = torch.zeros((len(arr), x.shape[1], n), device=device, dtype=torch.bool)
    with torch.inference_mode():
        out = model({"values": values, "masks": masks, "patch_is_target": target},
                    return_aux_outputs=True)
    return out["__call__:transformer_output"].cpu().half()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--timesfm-root", required=True); ap.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    ap.add_argument("--context", type=int, default=512); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); os.makedirs(a.output, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = load_model(a.timesfm_root, a.checkpoint, device)
    meta = json.load(open(a.input))
    for name, ds in meta.items():
        out = Path(a.output) / name; out.mkdir(exist_ok=True)
        for split in ("train", "test"):
            x = np.load(ds[f"{split}_x"], allow_pickle=True)
            y = np.load(ds[f"{split}_y"], allow_pickle=True)
            fs = []
            for start in range(0, len(x), a.batch_size):
                fs.extend(features_batch(model, x[start:start + a.batch_size], device, a.context))
            torch.save({"features": fs, "labels": y.tolist(), "channels": int(x[0].shape[0])}, out / f"{split}.pt")
        print(name, len(x), "->", out, flush=True)

if __name__ == "__main__": main()
