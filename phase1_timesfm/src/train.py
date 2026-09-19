import argparse, json, os, random, time, gc
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from heads import make_head

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def classes(labels): return {v:i for i,v in enumerate(sorted(set(labels)))}
def dataset_base(root, name):
    roots = root if isinstance(root, (list, tuple)) else [root]
    for r in roots:
        base = Path(r) / name
        if (base / "train.pt").exists() and (base / "test.pt").exists():
            return base
    raise FileNotFoundError(f"no complete feature pair for {name} in {roots}")

def shard_parts(shards_root, name, split):
    if not shards_root: return []
    d = Path(shards_root) / name / split
    return sorted(d.glob("part-*.pt")) if d.exists() else []

def shard_meta(shards_root, name, split):
    parts = shard_parts(shards_root, name, split)
    if not parts: raise FileNotFoundError(f"no shards for {name}/{split}")
    return json.load(open(parts[0].parent / "meta.json"))

def iter_shards(shards_root, name, split):
    for p in shard_parts(shards_root, name, split):
        z = torch.load(p, weights_only=False)
        x = torch.stack([t.float() for t in z["features"]])
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        yield x, z["labels"], z.get("channels")

def iter_source_shards(shards_root, name):
    # Coalesce small 16-example files into larger logical chunks. Large
    # variable-channel shards (notably PEMS-SF) exceed the byte cap and are
    # yielded individually, preserving the memory bound.
    xs, ys, channels, n_examples, n_bytes = [], [], None, 0, 0
    for split in ("train", "test"):
        for x, labels, c in iter_shards(shards_root, name, split):
            size = x.numel() * x.element_size()
            if xs and (n_examples + len(labels) > 128 or n_bytes + size > 512 * 1024 * 1024):
                yield torch.cat(xs, 0), sum(ys, []), channels
                xs, ys, channels, n_examples, n_bytes = [], [], None, 0, 0
            xs.append(x); ys.append(list(labels)); channels = c
            n_examples += len(labels); n_bytes += size
    if xs: yield torch.cat(xs, 0), sum(ys, []), channels

def shard_norm(shards_root, name, split):
    total = 0; s = None; ss = None
    for x, _, _ in iter_shards(shards_root, name, split):
        v = x.numel() // x.shape[-1]
        a = x.sum((0, 1, 2)); b = (x * x).sum((0, 1, 2))
        s = a if s is None else s + a; ss = b if ss is None else ss + b; total += v
    mean = s / max(1, total); var = (ss / max(1, total) - mean * mean).clamp_min(0)
    return mean, var.sqrt().clamp_min(1e-4)

def source_shard_norm(shards_root, name):
    total = 0; s = None; ss = None
    for x, _, _ in iter_source_shards(shards_root, name):
        v = x.numel() // x.shape[-1]
        a = x.sum((0, 1, 2)); b = (x * x).sum((0, 1, 2))
        s = a if s is None else s + a; ss = b if ss is None else ss + b; total += v
    mean = s / max(1, total); var = (ss / max(1, total) - mean * mean).clamp_min(0)
    return mean, var.sqrt().clamp_min(1e-4)

def load_split(root, name, split, aux_root=None, norm=None, use_all_train=False):
    base = dataset_base(root, name)
    z = torch.load(base / f"{split}.pt", weights_only=False)
    # Use one stable label map for the complete dataset. Building a map from
    # the current split alone silently changes class IDs when a rare class is
    # absent from validation/test.
    other_split = "test" if split == "train" else "train"
    oz = torch.load(base / f"{other_split}.pt", weights_only=False)
    if split == "train" and use_all_train:
        z = {"features": list(z["features"]) + list(oz["features"]), "labels": list(z["labels"]) + list(oz["labels"]), "channels": z.get("channels")}
    x = torch.stack([t.float() for t in z["features"]]); x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0); yraw = z["labels"]; mp = classes(list(yraw) + list(oz["labels"]))
    if aux_root:
        aux_base = dataset_base(aux_root, name)
        az = torch.load(aux_base / f"{split}.pt", weights_only=False)
        if split == "train" and use_all_train:
            at = torch.load(aux_base / "test.pt", weights_only=False)
            az = {"features": list(az["features"]) + list(at["features"])}
        ax = torch.stack([t.float() for t in az["features"]]); ax = torch.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
        if len(ax) != len(x): raise ValueError(f"feature row mismatch for {name}/{split}")
        # Contexts may produce different patch counts; align at the shared
        # channel level before concatenating the two frozen representations.
        x = torch.cat([x.mean(2, keepdim=True), ax.mean(2, keepdim=True)], dim=-1)
    if norm is not None:
        x = (x - norm[0]) / norm[1]
    return x, torch.tensor([mp[v] for v in yraw]), z["channels"], len(mp)

class Runner:
    def __init__(self, root, aux_root, names, variant, qpc, device, feature_norm=False, shards_root=None, label_info=None):
        self.root, self.aux_root, self.names, self.device = root, aux_root, names, device
        self.shards_root = shards_root
        self.label_info = label_info or {}
        self.batch_size = 64
        self.feature_norm = feature_norm; self.norms = {}
        if shards_root:
            first_x, _, _ = next(iter_shards(shards_root, names[0], "train")); self.dim = first_x.shape[-1]
            if feature_norm: self.norms[names[0]] = source_shard_norm(shards_root, names[0])
        else:
            first, _, c, k = load_split(root, names[0], "train", aux_root, use_all_train=True); self.dim = first.shape[-1]
            if feature_norm: self.norms[names[0]] = (first.mean((0,1,2)), first.std((0,1,2),unbiased=False).clamp_min(1e-4))
        self.head = make_head(variant, self.dim, qpc).to(device); self.query_dim = getattr(self.head, "query_dim", self.dim)
        self.adapters = nn.ParameterDict(); self.queries = nn.ParameterDict()
        self.qpc = qpc; self.variant = variant
        for n in names: self.add_dataset(n)
        self.adapters.to(device); self.queries.to(device)
        self.opt = torch.optim.AdamW(list(self.head.parameters()) + list(self.adapters.parameters()) + list(self.queries.parameters()), lr=2e-4, weight_decay=1e-4)
    def amp_context(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
    def add_dataset(self, name):
        is_source = name in self.names
        if self.shards_root and is_source:
            meta = shard_meta(self.shards_root, name, "train")
            c = int(meta["channels"])
            k = self.label_info[name][0] if name in self.label_info else len(set(y for _, ys, _ in iter_source_shards(self.shards_root, name) for y in ys))
            # The label vocabulary is obtained from the bounded shard stream.
            if self.feature_norm and name not in self.norms: self.norms[name] = source_shard_norm(self.shards_root, name)
        else:
            train_x, _, c, k = load_split(self.root, name, "train", self.aux_root, use_all_train=is_source)
            if self.feature_norm and name not in self.norms: self.norms[name] = (train_x.mean((0,1,2)), train_x.std((0,1,2),unbiased=False).clamp_min(1e-4))
        self.adapters[name] = nn.Parameter(torch.randn(c, self.dim, device=self.device) * 0.02)
        self.queries[name] = nn.Parameter(torch.randn(k * self.qpc, self.query_dim, device=self.device) * 0.02)
    def step_dataset(self, name, train=True, max_items=0, balanced=False, balance_alpha=1.0):
        if self.shards_root and train and name in self.names:
            # Stream official train and test shards as the full source corpus.
            # No complete dataset tensor is ever assembled in memory.
            if name in self.label_info:
                k, label_values, counts = self.label_info[name]
            else:
                all_labels = [y for _, ys, _ in iter_source_shards(self.shards_root, name) for y in ys]
                label_values = sorted(set(all_labels)); k = len(label_values)
                counts = torch.bincount(torch.tensor([label_values.index(y) for y in all_labels]), minlength=k).float().clamp_min(1)
            total = 0.; started = time.time()
            for part, (x, ys, _) in enumerate(iter_source_shards(self.shards_root, name)):
                mp = {v:i for i,v in enumerate(label_values)}
                y = torch.tensor([mp[v] for v in ys], dtype=torch.long)
                if self.norms.get(name) is not None: x = (x - self.norms[name][0]) / self.norms[name][1]
                # Bound the activation size using the actual stored tensor
                # shape. This is important for variable-channel/long UEA
                # datasets while retaining larger batches for small UCR sets.
                bs = max(1, min(self.batch_size, 32000000 // max(1, x.shape[1] * x.shape[2] * x.shape[3])))
                pin = x.numel() < 50000000
                if balanced:
                    weights = (int(counts.sum().item()) / (k * counts)).pow(balance_alpha)[y]
                    dl = DataLoader(TensorDataset(x, y, weights), batch_size=bs, shuffle=True, pin_memory=pin)
                else: dl = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, pin_memory=pin)
                for batch in dl:
                    xb, yb = batch[:2]; bw = batch[2].to(self.device) if len(batch) == 3 else None
                    xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                    with self.amp_context(): logits = self.head(xb, self.adapters[name], self.queries[name])
                    lv = nn.functional.cross_entropy(logits.float(), yb, reduction="none")
                    loss = (lv * bw).mean() if bw is not None else lv.mean()
                    self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0); self.opt.step()
                    total += float(loss.detach()) * len(yb)
                print(f"DATASET {name} part={part+1} elapsed={time.time()-started:.1f}s", flush=True)
                del dl, x, y
                gc.collect()
                if self.device.type == "cuda": torch.cuda.empty_cache()
            return total / max(1, int(counts.sum().item()))
        x, y, _, k = load_split(self.root, name, "train" if train else "test", self.aux_root, self.norms.get(name), use_all_train=(train and name in self.names))
        if max_items and len(y) > max_items:
            ix = torch.randperm(len(y))[:max_items]; x, y = x[ix], y[ix]
        bs = max(1, min(self.batch_size, 32000000 // max(1, x.shape[1] * x.shape[2] * x.shape[3])))
        if balanced and train:
            counts = torch.bincount(y, minlength=k).float().clamp_min(1)
            weights = (len(y) / (k * counts)).pow(balance_alpha)[y]
            dl = DataLoader(TensorDataset(x, y, weights), batch_size=bs, shuffle=True, pin_memory=x.numel() < 50000000)
        else:
            dl = DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=train, pin_memory=x.numel() < 50000000)
        total = 0.
        for batch in dl:
            xb, yb = batch[:2]
            bw = batch[2].to(self.device) if len(batch) == 3 else None
            xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            with self.amp_context(): logits = self.head(xb, self.adapters[name], self.queries[name])
            loss_vec = nn.functional.cross_entropy(logits.float(), yb, reduction="none")
            loss = (loss_vec * bw).mean() if bw is not None else loss_vec.mean()
            if train: self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0); self.opt.step()
            total += float(loss.detach()) * len(yb)
        return total / max(1, len(y))
    @torch.no_grad()
    def evaluate(self, name):
        x, y, _, _ = load_split(self.root, name, "test", self.aux_root, self.norms.get(name)); pred=[]
        for xb in DataLoader(x, batch_size=64, pin_memory=True):
            with self.amp_context(): logits = self.head(xb.to(self.device, non_blocking=True), self.adapters[name], self.queries[name])
            pred.extend(logits.argmax(-1).cpu().tolist())
        return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True,nargs="+"); ap.add_argument("--aux-features",default=""); ap.add_argument("--shards-root",default=""); ap.add_argument("--manifest",required=True); ap.add_argument("--eval-manifest",default=""); ap.add_argument("--variant",default="sda"); ap.add_argument("--qpc",type=int,default=4); ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--adapt-steps",type=int,default=100); ap.add_argument("--adapt-lr",type=float,default=1e-3); ap.add_argument("--adapt-head",action="store_true"); ap.add_argument("--prototype-init",action="store_true"); ap.add_argument("--balanced",action="store_true"); ap.add_argument("--balance-alpha",type=float,default=1.0); ap.add_argument("--feature-norm",action="store_true"); ap.add_argument("--holdout",default="ECG200"); ap.add_argument("--seed",type=int,default=7); ap.add_argument("--out",required=True); ap.add_argument("--resume-head",default="")
    a=ap.parse_args(); seed(a.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); manifest=json.load(open(a.manifest)); names=[n for n in manifest if n != a.holdout]; aux=a.aux_features or None
    label_info={}
    for n in names:
        vals=[]
        for split in ("train", "test"):
            vals.extend(np.load(manifest[n][f"{split}_y"], allow_pickle=True).tolist())
        uniq=sorted(set(vals)); counts=torch.tensor([sum(v == u for v in vals) for u in uniq], dtype=torch.float32).clamp_min(1)
        label_info[n]=(len(uniq), uniq, counts)
    r=Runner(a.features,aux,names,a.variant,a.qpc,dev,a.feature_norm,a.shards_root or None,label_info)
    if a.resume_head:
        payload=torch.load(a.resume_head, map_location="cpu", weights_only=False)
        r.head.load_state_dict(payload["head_state_dict"])
        print(f"RESUMED_HEAD path={a.resume_head}", flush=True)
    print(f"START variant={a.variant} datasets={len(names)} epochs={a.epochs} shards={bool(a.shards_root)} device={dev}", flush=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_ckpt_dir = out_path.parent / (out_path.stem + "_tasks")
    def save_head_checkpoint(path, task=None):
        payload = {
            "format": "timesfm_classifier_head_v1",
            "variant": a.variant,
            "dim": r.dim,
            "qpc": r.qpc,
            "head_config": {"variant": a.variant, "dim": r.dim, "qpc": r.qpc},
            "head_state_dict": {k: v.detach().cpu() for k, v in r.head.state_dict().items()},
        }
        if task is not None:
            payload.update({
                "task": task,
                "adapter": r.adapters[task].detach().cpu(),
                "queries": r.queries[task].detach().cpu(),
            })
        torch.save(payload, path)
        print(f"CHECKPOINT_SAVED path={path} task={task or 'source_head'} bytes={path.stat().st_size}", flush=True)
    run_start=time.time()
    for ep in range(a.epochs):
        random.shuffle(names); losses=[]
        for di,n in enumerate(names,1):
            losses.append(r.step_dataset(n, True, balanced=a.balanced, balance_alpha=a.balance_alpha))
            elapsed=time.time()-run_start; done=ep*len(names)+di; total=a.epochs*len(names); eta=elapsed/max(1,done)*(total-done)
            print(f"PROGRESS epoch={ep+1}/{a.epochs} dataset={di}/{len(names)} name={n} loss={losses[-1]:.5f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
        print(f"EPOCH {ep+1}/{a.epochs} mean_loss={sum(losses)/len(losses):.5f}", flush=True)
    # Persist the trained classifier head before any evaluation/adaptation.
    # The frozen TimesFM3 backbone and cached features are deliberately absent.
    source_ckpt = out_path.with_suffix(".source_head.pt")
    save_head_checkpoint(source_ckpt)
    # Few-shot adaptation on one or many strictly unseen datasets. The
    # multi-eval path is used for the FlaMinGo comparison: all published
    # benchmark datasets are absent from --manifest during shared training.
    if a.eval_manifest:
        eval_names=list(json.load(open(a.eval_manifest)).keys())
    else:
        eval_names=[a.holdout]
    shared=list(r.head.parameters()); [p.requires_grad_(False) for p in shared]
    eval_metrics={}
    for eval_name in eval_names:
        r.add_dataset(eval_name)
        adapt_opt=torch.optim.AdamW([r.adapters[eval_name], r.queries[eval_name]], lr=a.adapt_lr)
        adapt_x, adapt_y, _, _ = load_split(a.features, eval_name, "train", aux, r.norms.get(eval_name))
        if a.prototype_init:
            with torch.no_grad():
                r.adapters[eval_name].zero_()
                nclasses=int(adapt_y.max().item())+1
                pooled=(adapt_x.to(dev)+r.adapters[eval_name][None,:,None,:]).mean((1,2))
                z=r.head.out(r.head.proj(pooled)) if a.variant == "linear" else pooled
                proto=torch.stack([z[adapt_y.to(dev)==j].mean(0) for j in range(nclasses)])
                r.queries[eval_name].copy_(proto.repeat_interleave(r.qpc,0))
        for _ in range(a.adapt_steps):
            ix=torch.randperm(len(adapt_y))[:min(64,len(adapt_y))]
            with r.amp_context(): logits=r.head(adapt_x[ix].to(dev, non_blocking=True),r.adapters[eval_name],r.queries[eval_name])
            if a.balanced:
                yy=adapt_y[ix].to(dev); counts=torch.bincount(adapt_y, minlength=logits.shape[-1]).float().clamp_min(1).to(dev); cw=(len(adapt_y)/(logits.shape[-1]*counts)).pow(a.balance_alpha).to(dtype=logits.dtype); loss=nn.functional.cross_entropy(logits,yy,weight=cw)
            else: loss=nn.functional.cross_entropy(logits,adapt_y[ix].to(dev))
            adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
        task_ckpt_dir.mkdir(parents=True, exist_ok=True)
        task_ckpt = task_ckpt_dir / f"{eval_name}.pt"
        # At this point only the task adapter and label queries were updated;
        # save them together with the frozen head before test evaluation.
        save_head_checkpoint(task_ckpt, task=eval_name)
        eval_metrics[eval_name]=r.evaluate(eval_name)
        print("EVAL", eval_name, eval_metrics[eval_name], flush=True)
    result={"variant":a.variant,"holdout":a.holdout,"eval_datasets":eval_names,"metrics":eval_metrics if a.eval_manifest else eval_metrics[a.holdout],"seed":a.seed,"train_datasets":names,"head_params":sum(p.numel() for p in r.head.parameters()),"backbone":"TimesFM3 frozen; cached TimesFM3 variate-attention features"}
    # Keep the public result checkpoint head-only; per-task adapted weights are
    # stored separately in the *_tasks directory.
    result["checkpoint"] = str(source_ckpt)
    json.dump(result,open(out_path,"w"),indent=2); print(json.dumps(result),flush=True)
if __name__ == "__main__": main()
