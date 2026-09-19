import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from heads import make_head

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def classes(labels): return {v:i for i,v in enumerate(sorted(set(labels)))}
def load_split(root, name, split, aux_root=None, norm=None):
    z = torch.load(Path(root) / name / f"{split}.pt", weights_only=False)
    x = torch.stack([t.float() for t in z["features"]]); x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0); yraw = z["labels"]; mp = classes(yraw)
    if aux_root:
        az = torch.load(Path(aux_root) / name / f"{split}.pt", weights_only=False)
        ax = torch.stack([t.float() for t in az["features"]]); ax = torch.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
        if len(ax) != len(x): raise ValueError(f"feature row mismatch for {name}/{split}")
        # Contexts may produce different patch counts; align at the shared
        # channel level before concatenating the two frozen representations.
        x = torch.cat([x.mean(2, keepdim=True), ax.mean(2, keepdim=True)], dim=-1)
    if norm is not None:
        x = (x - norm[0]) / norm[1]
    return x, torch.tensor([mp[v] for v in yraw]), z["channels"], len(mp)

class Runner:
    def __init__(self, root, aux_root, names, variant, qpc, device, feature_norm=False):
        self.root, self.aux_root, self.names, self.device = root, aux_root, names, device
        self.feature_norm = feature_norm; self.norms = {}
        first, _, c, k = load_split(root, names[0], "train", aux_root); self.dim = first.shape[-1]
        if feature_norm: self.norms[names[0]] = (first.mean((0,1,2)), first.std((0,1,2),unbiased=False).clamp_min(1e-4))
        self.head = make_head(variant, self.dim, qpc).to(device); self.query_dim = getattr(self.head, "query_dim", self.dim)
        self.adapters = nn.ParameterDict(); self.queries = nn.ParameterDict()
        self.qpc = qpc; self.variant = variant
        for n in names: self.add_dataset(n)
        self.adapters.to(device); self.queries.to(device)
        self.opt = torch.optim.AdamW(list(self.head.parameters()) + list(self.adapters.parameters()) + list(self.queries.parameters()), lr=2e-4, weight_decay=1e-4)
    def add_dataset(self, name):
        train_x, _, c, k = load_split(self.root, name, "train", self.aux_root)
        if self.feature_norm and name not in self.norms: self.norms[name] = (train_x.mean((0,1,2)), train_x.std((0,1,2),unbiased=False).clamp_min(1e-4))
        self.adapters[name] = nn.Parameter(torch.randn(c, self.dim, device=self.device) * 0.02)
        self.queries[name] = nn.Parameter(torch.randn(k * self.qpc, self.query_dim, device=self.device) * 0.02)
    def step_dataset(self, name, train=True, max_items=0, balanced=False, balance_alpha=1.0):
        x, y, _, k = load_split(self.root, name, "train" if train else "test", self.aux_root, self.norms.get(name))
        if max_items and len(y) > max_items:
            ix = torch.randperm(len(y))[:max_items]; x, y = x[ix], y[ix]
        if balanced and train:
            counts = torch.bincount(y, minlength=k).float().clamp_min(1)
            weights = (len(y) / (k * counts)).pow(balance_alpha)[y]
            dl = DataLoader(TensorDataset(x, y, weights), batch_size=16, shuffle=True)
        else:
            dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=train)
        total = 0.
        for batch in dl:
            xb, yb = batch[:2]
            bw = batch[2].to(self.device) if len(batch) == 3 else None
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.head(xb, self.adapters[name], self.queries[name])
            loss_vec = nn.functional.cross_entropy(logits, yb, reduction="none")
            loss = (loss_vec * bw).mean() if bw is not None else loss_vec.mean()
            if train: self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0); self.opt.step()
            total += float(loss.detach()) * len(yb)
        return total / max(1, len(y))
    @torch.no_grad()
    def evaluate(self, name):
        x, y, _, _ = load_split(self.root, name, "test", self.aux_root, self.norms.get(name)); pred=[]
        for xb in DataLoader(x, batch_size=32): pred.extend(self.head(xb.to(self.device), self.adapters[name], self.queries[name]).argmax(-1).cpu().tolist())
        return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--aux-features",default=""); ap.add_argument("--manifest",required=True); ap.add_argument("--variant",default="sda"); ap.add_argument("--qpc",type=int,default=4); ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--adapt-steps",type=int,default=100); ap.add_argument("--adapt-lr",type=float,default=1e-3); ap.add_argument("--adapt-head",action="store_true"); ap.add_argument("--prototype-init",action="store_true"); ap.add_argument("--balanced",action="store_true"); ap.add_argument("--balance-alpha",type=float,default=1.0); ap.add_argument("--feature-norm",action="store_true"); ap.add_argument("--holdout",default="ECG200"); ap.add_argument("--seed",type=int,default=7); ap.add_argument("--out",required=True)
    a=ap.parse_args(); seed(a.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); manifest=json.load(open(a.manifest)); names=[n for n in manifest if n != a.holdout]; aux=a.aux_features or None
    r=Runner(a.features,aux,names,a.variant,a.qpc,dev,a.feature_norm)
    for ep in range(a.epochs):
        random.shuffle(names); losses=[r.step_dataset(n, True, balanced=a.balanced, balance_alpha=a.balance_alpha) for n in names]
        if ep % 5 == 0: print(a.variant, ep, sum(losses)/len(losses), flush=True)
    # Few-shot adaptation on the held-out dataset, freezing shared head and training only new tokens.
    r.add_dataset(a.holdout)
    if a.adapt_head:
        adapt_opt=torch.optim.AdamW(list(r.head.parameters())+[r.adapters[a.holdout], r.queries[a.holdout]], lr=a.adapt_lr)
    else:
        shared=list(r.head.parameters()); [p.requires_grad_(False) for p in shared]
        adapt_opt=torch.optim.AdamW([r.adapters[a.holdout], r.queries[a.holdout]], lr=a.adapt_lr)
    adapt_x, adapt_y, _, _ = load_split(a.features, a.holdout, "train", aux, r.norms.get(a.holdout))
    if a.prototype_init and a.variant == "linear":
        with torch.no_grad():
            r.adapters[a.holdout].zero_()
            pooled=(adapt_x.to(dev)+r.adapters[a.holdout][None,:,None,:]).mean((1,2))
            z=r.head.out(r.head.proj(pooled))
            nclasses=int(adapt_y.max().item())+1
            proto=torch.stack([z[adapt_y.to(dev)==j].mean(0) for j in range(nclasses)])
            r.queries[a.holdout].copy_(proto.repeat_interleave(r.qpc,0))
    for _ in range(a.adapt_steps):
        x,y=adapt_x, adapt_y; ix=torch.randperm(len(y))[:min(64,len(y))]
        logits=r.head(x[ix].to(dev),r.adapters[a.holdout],r.queries[a.holdout]);
        if a.balanced:
            yy=y[ix].to(dev); counts=torch.bincount(y, minlength=logits.shape[-1]).float().clamp_min(1).to(dev); cw=(len(y)/(logits.shape[-1]*counts)).pow(a.balance_alpha); loss=nn.functional.cross_entropy(logits,yy,weight=cw)
        else: loss=nn.functional.cross_entropy(logits,y[ix].to(dev))
        adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
    result={"variant":a.variant,"holdout":a.holdout,"seed":a.seed,"metrics":r.evaluate(a.holdout),"train_datasets":names,"head_params":sum(p.numel() for p in r.head.parameters()),"backbone":"TimesFM3 frozen; cached transformer features"}
    out_path = Path(a.out); out_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"variant": a.variant, "dim": r.dim, "qpc": r.qpc,
                "head_state_dict": r.head.state_dict(),
                "train_adapters": {k: v.detach().cpu() for k,v in r.adapters.items()},
                "train_queries": {k: v.detach().cpu() for k,v in r.queries.items()},
                "heldout_adapter": r.adapters[a.holdout].detach().cpu(),
                "heldout_queries": r.queries[a.holdout].detach().cpu()},
               out_path.with_suffix(".pt"))
    result["checkpoint"] = str(out_path.with_suffix(".pt"))
    json.dump(result,open(out_path,"w"),indent=2); print(json.dumps(result),flush=True)
if __name__ == "__main__": main()
