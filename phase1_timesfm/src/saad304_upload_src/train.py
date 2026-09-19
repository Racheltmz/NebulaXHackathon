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
def load_split(root, name, split):
    z = torch.load(Path(root) / name / f"{split}.pt", weights_only=False)
    x = torch.stack([t.float() for t in z["features"]]); yraw = z["labels"]; mp = classes(yraw)
    return x, torch.tensor([mp[v] for v in yraw]), z["channels"], len(mp)

class Runner:
    def __init__(self, root, names, variant, qpc, device):
        self.root, self.names, self.device = root, names, device
        first, _, c, k = load_split(root, names[0], "train"); self.dim = first.shape[-1]
        self.head = make_head(variant, self.dim, qpc).to(device)
        self.adapters = nn.ParameterDict(); self.queries = nn.ParameterDict()
        self.qpc = qpc; self.variant = variant
        for n in names: self.add_dataset(n)
        self.adapters.to(device); self.queries.to(device)
        self.opt = torch.optim.AdamW(list(self.head.parameters()) + list(self.adapters.parameters()) + list(self.queries.parameters()), lr=2e-4, weight_decay=1e-4)
    def add_dataset(self, name):
        _, _, c, k = load_split(self.root, name, "train")
        self.adapters[name] = nn.Parameter(torch.randn(c, self.dim, device=self.device) * 0.02)
        self.queries[name] = nn.Parameter(torch.randn(k * self.qpc, self.dim, device=self.device) * 0.02)
    def step_dataset(self, name, train=True, max_items=0):
        x, y, _, k = load_split(self.root, name, "train" if train else "test")
        if max_items and len(y) > max_items:
            ix = torch.randperm(len(y))[:max_items]; x, y = x[ix], y[ix]
        dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=train)
        total = 0.
        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.head(xb, self.adapters[name], self.queries[name])
            loss = nn.functional.cross_entropy(logits, yb)
            if train: self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0); self.opt.step()
            total += float(loss.detach()) * len(yb)
        return total / max(1, len(y))
    @torch.no_grad()
    def evaluate(self, name):
        x, y, _, _ = load_split(self.root, name, "test"); pred=[]
        for xb in DataLoader(x, batch_size=32): pred.extend(self.head(xb.to(self.device), self.adapters[name], self.queries[name]).argmax(-1).cpu().tolist())
        return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--manifest",required=True); ap.add_argument("--variant",default="sda"); ap.add_argument("--qpc",type=int,default=4); ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--holdout",default="ECG200"); ap.add_argument("--seed",type=int,default=7); ap.add_argument("--out",required=True)
    a=ap.parse_args(); seed(a.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); manifest=json.load(open(a.manifest)); names=[n for n in manifest if n != a.holdout]
    r=Runner(a.features,names,a.variant,a.qpc,dev)
    for ep in range(a.epochs):
        random.shuffle(names); losses=[r.step_dataset(n, True) for n in names]
        if ep % 5 == 0: print(a.variant, ep, sum(losses)/len(losses), flush=True)
    # Few-shot adaptation on the held-out dataset, freezing shared head and training only new tokens.
    r.add_dataset(a.holdout); shared=list(r.head.parameters()); [p.requires_grad_(False) for p in shared]
    adapt_opt=torch.optim.AdamW([r.adapters[a.holdout], r.queries[a.holdout]], lr=1e-3)
    for _ in range(20):
        x,y,_,_=load_split(a.features,a.holdout,"train"); ix=torch.randperm(len(y))[:min(64,len(y))]
        logits=r.head(x[ix].to(dev),r.adapters[a.holdout],r.queries[a.holdout]); loss=nn.functional.cross_entropy(logits,y[ix].to(dev)); adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
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
