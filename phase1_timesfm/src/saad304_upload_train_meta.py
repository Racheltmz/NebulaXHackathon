import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.metrics import accuracy_score,f1_score
from heads import make_head

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def load(root,name,split):
    z=torch.load(Path(root)/name/f'{split}.pt',weights_only=False)
    x=torch.stack([t.float() for t in z['features']]); x=torch.nan_to_num(x)
    vals=sorted(set(z['labels'])); mp={v:i for i,v in enumerate(vals)}
    y=torch.tensor([mp[v] for v in z['labels']]); return x,y,int(x.shape[1]),len(vals)
def balanced_loss(logits,y):
    k=logits.shape[-1]; c=torch.bincount(y,minlength=k).float().clamp_min(1); w=(len(y)/(k*c)).to(logits.device)
    return nn.functional.cross_entropy(logits,y.to(logits.device),weight=w)
def episode_split(x,y):
    sup=[]; qry=[]
    for c in torch.unique(y):
        ii=torch.where(y==c)[0][torch.randperm(int((y==c).sum()))]; n=max(1,int(.5*len(ii))); sup.append(ii[:n]); qry.append(ii[n:])
    si=torch.cat(sup); qi=torch.cat([v for v in qry if len(v)]); return x[si],y[si],x[qi],y[qi]
def adapt(head,x,y,adapter,queries,steps,lr):
    adapter=adapter.detach().requires_grad_(); queries=queries.detach().requires_grad_()
    for _ in range(steps):
        loss=balanced_loss(head(x,adapter,queries),y); ga,gq=torch.autograd.grad(loss,(adapter,queries))
        adapter=(adapter-lr*ga).detach().requires_grad_(); queries=(queries-lr*gq).detach().requires_grad_()
    return adapter.detach(),queries.detach()

ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--holdout',required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--inner',type=int,default=3); ap.add_argument('--qpc',type=int,default=4); a=ap.parse_args()
seed(7); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); names=[n for n in json.load(open(a.manifest)) if n!=a.holdout]
first,_,c,k=load(a.features,names[0],'train'); head=make_head('linear',first.shape[-1],a.qpc).to(dev); opt=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=1e-4)
for ep in range(a.epochs):
    random.shuffle(names)
    for name in names:
        x,y,c,k=load(a.features,name,'train'); xs,ys,xq,yq=episode_split(x,y)
        ad=torch.randn(c,first.shape[-1],device=dev)*.02; qu=torch.randn(k*a.qpc,first.shape[-1],device=dev)*.02
        ad,qu=adapt(head,xs.to(dev),ys.to(dev),ad,qu,a.inner,1e-3); logits=head(xq.to(dev),ad,qu); loss=balanced_loss(logits,yq.to(dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
    print('meta',ep,flush=True)
x,y,c,k=load(a.features,a.holdout,'train'); xt,yt,_,_=load(a.features,a.holdout,'test'); ad=torch.randn(c,first.shape[-1],device=dev)*.02; qu=torch.randn(k*a.qpc,first.shape[-1],device=dev)*.02
ad,qu=adapt(head,x.to(dev),y.to(dev),ad,qu,max(20,a.inner*20),1e-3)
with torch.no_grad(): pred=head(xt.to(dev),ad,qu).argmax(-1).cpu()
metrics={'accuracy':float(accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'head':head.state_dict(),'metrics':metrics},out.with_suffix('.pt'))
json.dump({'variant':'episodic_meta','inner':a.inner,'metrics':metrics,'checkpoint':str(out.with_suffix('.pt'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
