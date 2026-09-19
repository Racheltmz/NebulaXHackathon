"""Minimal PS3 fine-tuning entry point over cached frozen-backbone features.

Input is a torch file containing ``features`` (list of [C,L,D] tensors) and
``labels``. Classification trains new channel/class tokens; regression trains
the shared scalar probe plus new channel tokens. Raw-data parsing is deliberately
kept in ps3_adapter.py because each Info Kit has a different schema.
"""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from heads import make_head

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True); ap.add_argument("--mode", choices=["classification", "regression"], default="classification"); ap.add_argument("--classes", type=int, default=2); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.load(a.features, weights_only=False); xs = [x.float() for x in z["features"]]; ys = torch.tensor(z["labels"])
    dim = xs[0].shape[-1]; c = xs[0].shape[0]; qpc = 4
    head = make_head("regression" if a.mode == "regression" else "sda", dim, qpc).to(dev)
    ce = nn.Parameter(torch.zeros(c, dim, device=dev)); nn.init.normal_(ce, std=.02)
    q = nn.Parameter(torch.randn(a.classes * qpc, dim, device=dev) * .02)
    params = list(head.parameters()) + [ce] + ([] if a.mode == "regression" else [q])
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    for _ in range(a.epochs):
        for x, y in zip(xs, ys):
            x = x[None].to(dev); y = y.to(dev).reshape(1)
            pred = head(x, ce, None if a.mode == "regression" else q)
            loss = (pred.reshape(-1) - y.float()).square().mean() if a.mode == "regression" else nn.functional.cross_entropy(pred, y.long())
            opt.zero_grad(); loss.backward(); opt.step()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "channel_embeddings": ce.detach().cpu(), "label_queries": q.detach().cpu(), "mode": a.mode}, a.out)
    json.dump({"mode": a.mode, "n": len(xs), "out": a.out}, open(str(a.out) + ".json", "w"), indent=2)

if __name__ == "__main__": main()
