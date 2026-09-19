"""Convert cached TimesFM features into bounded-size dataset shards.

The original extractor stores one large torch file per split.  This utility
keeps the original cache intact and writes chunk files so training can load
one bounded chunk at a time, including for very large multivariate UEA sets.
"""
import argparse, json
from pathlib import Path
import torch


def find_base(roots, name):
    for root in roots:
        base = Path(root) / name
        if (base / "train.pt").exists() and (base / "test.pt").exists():
            return base
    raise FileNotFoundError(name)


def convert(base, out_root, name, split, chunk_size):
    src = torch.load(base / f"{split}.pt", weights_only=False)
    features, labels = src["features"], src["labels"]
    actual_channels = int(features[0].shape[0])
    out = Path(out_root) / name / split
    out.mkdir(parents=True, exist_ok=True)
    expected = (len(features) + chunk_size - 1) // chunk_size
    meta_path = out / "meta.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if (existing.get("count") == len(features) and existing.get("parts") == expected
            and existing.get("channels") == actual_channels):
        return existing
    # Remove stale chunks when rerunning a dataset.
    for p in out.glob("part-*.pt"):
        p.unlink()
    for start in range(0, len(features), chunk_size):
        stop = min(len(features), start + chunk_size)
        torch.save({"features": list(features[start:stop]),
                    "labels": list(labels[start:stop]),
                    "channels": actual_channels},
                   out / f"part-{start // chunk_size:05d}.pt")
    meta = {"count": len(features), "channels": actual_channels,
            "parts": (len(features) + chunk_size - 1) // chunk_size,
            "chunk_size": chunk_size}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--array-index", type=int, default=-1)
    ap.add_argument("--array-size", type=int, default=1)
    a = ap.parse_args()
    manifest = json.load(open(a.manifest))
    names = [a.dataset] if a.dataset else list(manifest)
    if a.array_index >= 0:
        names = [n for i, n in enumerate(names) if i % a.array_size == a.array_index]
    for i, name in enumerate(names, 1):
        base = find_base(a.features, name)
        print(f"SHARD_START {i}/{len(names)} {name}", flush=True)
        for split in ("train", "test"):
            meta = convert(base, a.out, name, split, a.chunk_size)
            print(f"SHARD_DONE {name} {split} {meta['count']} examples "
                  f"{meta['parts']} parts", flush=True)


if __name__ == "__main__":
    main()
