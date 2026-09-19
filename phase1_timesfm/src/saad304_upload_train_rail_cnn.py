import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score,f1_score
import joblib

ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--width',type=int,default=32); ap.add_argument('--kernel',type=int,default=7); ap.add_argument('--epochs',type=int,default=160); a=ap.parse_args()
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y0=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yt0=np.load(a.arrays+'_test_y.npy',allow_pickle=True); classes=np.unique(y0); mp={c:i for i,c in enumerate(classes)}; y=np.array([mp[c] for c in y0]); yt=np.array([mp[c] for c in yt0])
ix=np.linspace(0,x.shape[-1]-1,1000).astype(int); x=x[:,:,ix]; xt=xt[:,:,ix]; mu=x.mean((0,2),keepdims=True); sd=np.maximum(x.std((0,2),keepdims=True),1e-5); x=(x-mu)/sd; xt=(xt-mu)/sd
class Block(nn.Module):
    def __init__(self,w,k,d):
        super().__init__(); p=(k-1)*d//2; self.net=nn.Sequential(nn.Conv1d(w,w,k,padding=p,dilation=d),nn.BatchNorm1d(w),nn.GELU(),nn.Dropout(.1))
    def forward(self,z): return z+self.net(z)
class Net(nn.Module):
    def __init__(self,c,w,k):
        super().__init__(); self.stem=nn.Sequential(nn.Conv1d(c,w,k,padding=k//2),nn.BatchNorm1d(w),nn.GELU(),nn.MaxPool1d(4))
        self.blocks=nn.Sequential(Block(w,k,1),Block(w,k,2),Block(w,k,4),Block(w,k,8)); self.out=nn.Sequential(nn.LayerNorm(2*w),nn.Linear(2*w,w),nn.GELU(),nn.Dropout(.2),nn.Linear(w,len(classes)))
    def forward(self,z):
        h=self.blocks(self.stem(z)); return self.out(torch.cat([h.mean(-1),h.amax(-1)],-1))
torch.manual_seed(7); model=Net(x.shape[1],a.width,a.kernel).to(dev); tx=torch.from_numpy(x); ty=torch.from_numpy(y); weights=1/torch.bincount(ty).float(); sampler=WeightedRandomSampler(weights[ty],len(ty),replacement=True); dl=DataLoader(TensorDataset(tx,ty),batch_size=32,sampler=sampler); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-4); lossfn=nn.CrossEntropyLoss()
model.train()
for _ in range(a.epochs):
    for xb,yb in dl:
        pred=model(xb.to(dev)); loss=lossfn(pred,yb.to(dev)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); opt.step()
model.eval();
with torch.no_grad(): p=model(torch.from_numpy(xt).to(dev)).argmax(-1).cpu().numpy()
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,p)),'macro_f1':float(f1_score(yt,p,average='macro'))}; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'metrics':metrics},out.with_suffix('.pt')); json.dump({'model':'multiscale_cnn','width':a.width,'kernel':a.kernel,'epochs':a.epochs,'metrics':metrics,'checkpoint':str(out.with_suffix('.pt'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
