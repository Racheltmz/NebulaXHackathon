import argparse,json,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from sklearn.metrics import accuracy_score,f1_score

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def load(root,name,split):
    z=torch.load(Path(root)/name/f'{split}.pt',weights_only=False)
    x=torch.stack([t.float() for t in z['features']]); x=torch.nan_to_num(x)
    vals=sorted(set(z['labels'])); mp={v:i for i,v in enumerate(vals)}
    y=torch.tensor([mp[v] for v in z['labels']]); return x,y
def embed_features(x, mode):
    # Preserve channel structure while making a fixed task-level representation.
    if mode=='mean': return x.mean((1,2))
    if mode=='stats': return torch.cat([x.mean((1,2)),x.std((1,2),unbiased=False)],-1)
    if mode=='patch': return torch.cat([x.mean((1,2)),x.mean(1).std(1,unbiased=False)],-1)
    raise ValueError(mode)
def episode(x,y):
    si=[]; qi=[]
    for c in torch.unique(y):
        ii=torch.where(y==c)[0][torch.randperm(int((y==c).sum()))]
        ns=max(1,min(len(ii)-1,int(len(ii)*.5))) if len(ii)>1 else 1
        si.append(ii[:ns]); qi.append(ii[ns:])
    q=[z for z in qi if len(z)]
    return x[torch.cat(si)],y[torch.cat(si)],x[torch.cat(q)],y[torch.cat(q)]
class Metric(nn.Module):
    def __init__(self,d,emb,temperature):
        super().__init__(); self.temperature=temperature
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,512),nn.GELU(),nn.Linear(512,emb),nn.LayerNorm(emb))
    def forward(self,x): return nn.functional.normalize(self.net(x),dim=-1)
def logits(zs,ys,zq,temp):
    cs=[]
    for c in torch.unique(ys): cs.append(zs[ys==c].mean(0))
    c=torch.stack(cs); c=nn.functional.normalize(c,dim=-1)
    return zq@c.T/temp, torch.unique(ys)
ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--holdout',required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--mode',default='stats'); ap.add_argument('--embed',type=int,default=128); ap.add_argument('--temperature',type=float,default=.1); ap.add_argument('--episodes',type=int,default=2); a=ap.parse_args()
seed(7); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); names=[n for n in json.load(open(a.manifest)) if n!=a.holdout]
x0,y0=load(a.features,names[0],'train'); d=embed_features(x0,a.mode).shape[-1]; model=Metric(d,a.embed,a.temperature).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
for ep in range(a.epochs):
    random.shuffle(names)
    for name in names:
        x,y=load(a.features,name,'train');
        for _ in range(a.episodes):
            xs,ys,xq,yq=episode(x,y); zs=model(embed_features(xs,a.mode).to(dev)); zq=model(embed_features(xq,a.mode).to(dev)); lg,classes=logits(zs,ys.to(dev),zq,a.temperature); target=torch.tensor([int((classes==v).nonzero()[0]) for v in yq],device=dev); loss=nn.functional.cross_entropy(lg,target)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    print('proto',ep,flush=True)
x,y=load(a.features,a.holdout,'train'); xt,yt=load(a.features,a.holdout,'test')
with torch.no_grad(): zs=model(embed_features(x,a.mode).to(dev)); zt=model(embed_features(xt,a.mode).to(dev)); lg,classes=logits(zs,y.to(dev),zt,a.temperature); pred=classes[lg.argmax(-1)].cpu()
metrics={'accuracy':float(accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro'))}; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'metrics':metrics},out.with_suffix('.pt')); json.dump({'variant':'proto_metric','mode':a.mode,'embed':a.embed,'temperature':a.temperature,'metrics':metrics,'checkpoint':str(out.with_suffix('.pt'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
