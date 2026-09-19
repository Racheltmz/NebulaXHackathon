import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_percentage_error
from heads import make_head

def raw_summary(arr):
    rows = []
    for z in arr:
        original_len = np.asarray(z).shape[-1]
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1: z = z[None, :]
        # Keep feature cost bounded for the long SHM traces while retaining
        # low-frequency and defect-band information.
        if z.shape[-1] > 4096:
            ix = np.linspace(0, z.shape[-1]-1, 4096).astype(np.int64)
            z = z[:, ix]
        d = np.diff(z, axis=-1)
        q = np.quantile(z, [.05,.1,.25,.5,.75,.9,.95], axis=-1).reshape(-1)
        mean, std = z.mean(-1), z.std(-1)
        rms, peak = np.sqrt((z*z).mean(-1)), np.abs(z).max(-1)
        trend = np.polyfit(np.arange(z.shape[-1], dtype=np.float32), z.T, 1)[0]
        spec = np.abs(np.fft.rfft(z, axis=-1)) ** 2
        n = spec.shape[-1]
        bands = np.stack([spec[:, :max(2,n//16)].mean(-1),
                          spec[:, n//16:max(2,n//8)].mean(-1),
                          spec[:, n//8:max(3,n//4)].mean(-1),
                          spec[:, n//4:].mean(-1)], axis=1).reshape(-1)
        centroid = (spec * np.arange(n, dtype=np.float32)).sum(-1) / (spec.sum(-1)+1e-6)
        edge = np.concatenate([z[:,-1]-z[:,0], (z[:,-1]-z[:,0])/(np.abs(z).mean(-1)+1e-5)])
        duration = np.array([np.log1p(original_len)], dtype=np.float32)
        rows.append(np.concatenate([mean, std, rms, peak, q, d.std(-1),
                                    d.mean(-1), trend, bands, centroid, edge, duration]))
    return np.stack(rows).astype(np.float32)

def raw_resample(arr, length=1024):
    out=[]
    for z in arr:
        z=np.asarray(z,dtype=np.float32)
        if z.ndim==1: z=z[None,:]
        old=np.linspace(0,1,z.shape[-1],dtype=np.float32); new=np.linspace(0,1,length,dtype=np.float32)
        out.append(np.stack([np.interp(new,old,ch) for ch in z]))
    return np.stack(out).astype(np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--features',required=True); ap.add_argument('--raw'); ap.add_argument('--raw-only',action='store_true'); ap.add_argument('--task',choices=['classification','regression'],required=True); ap.add_argument('--variant',default='pool'); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--init'); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=7)
    a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=Path(a.data); fs=Path(a.features); train=torch.load(fs/'train.pt',weights_only=False); test=torch.load(fs/'test.pt',weights_only=False)
    x=torch.stack([z.float() for z in train['features']]); xv=torch.stack([z.float() for z in test['features']]); raw=np.asarray(train['labels']); rawv=np.asarray(test['labels'])
    raw_train = raw_test = raw_seq_train = raw_seq_test = None
    if a.raw:
        rtr=np.load(a.raw+'_train_x.npy', allow_pickle=True); rte=np.load(a.raw+'_test_x.npy', allow_pickle=True)
        raw_train = torch.from_numpy(raw_summary(rtr)); raw_test = torch.from_numpy(raw_summary(rte))
        raw_seq_train = torch.from_numpy(raw_resample(rtr)); raw_seq_test = torch.from_numpy(raw_resample(rte))
    if a.task == 'classification':
        vals=sorted(set(raw.tolist())); mp={v:i for i,v in enumerate(vals)}; y=torch.tensor([mp[v] for v in raw]); yv=torch.tensor([mp.get(v,0) for v in rawv]); k=len(vals); head=make_head(a.variant,x.shape[-1],4).to(dev); q=torch.nn.Parameter(torch.randn(k*4,x.shape[-1],device=dev)*.02); loss_fn=nn.CrossEntropyLoss(weight=(torch.bincount(y).float().sum()/torch.clamp(torch.bincount(y).float(),min=1)).to(dev))
    else:
        y=torch.tensor(raw,dtype=torch.float32); yv=torch.tensor(rawv,dtype=torch.float32); k=1; head=make_head('regression',x.shape[-1],4).to(dev); q=None
        y_log=torch.log(y.clamp_min(1e-5)); y_mean, y_std = y_log.mean(), y_log.std().clamp_min(1e-4); y_scaled=(y_log-y_mean)/y_std; yv_scaled=(torch.log(yv.clamp_min(1e-5))-y_mean)/y_std; loss_fn=nn.SmoothL1Loss()
    if a.init:
        ck=torch.load(a.init,weights_only=False); head.load_state_dict(ck['head_state_dict'],strict=False)
    c=train['features'][0].shape[0]; ce=torch.nn.Parameter(torch.randn(c,x.shape[-1],device=dev)*.02)
    if raw_train is not None:
        rmean, rstd = raw_train.mean(0), raw_train.std(0).clamp_min(1e-5)
        raw_train = (raw_train-rmean)/rstd; raw_test = (raw_test-rmean)/rstd
        stats_head = nn.Sequential(nn.LayerNorm(raw_train.shape[1]), nn.Linear(raw_train.shape[1],256), nn.GELU(), nn.Dropout(.1), nn.Linear(256,k)).to(dev)
        class SignalHead(nn.Module):
            def __init__(self, channels, outputs):
                super().__init__()
                self.body=nn.Sequential(nn.Conv1d(channels,32,9,padding=4), nn.GELU(), nn.MaxPool1d(4),
                                        nn.Conv1d(32,64,9,padding=4), nn.GELU())
                self.out=nn.Linear(128,outputs)
            def forward(self, signal):
                h=self.body(signal)
                return self.out(torch.cat([h.mean(-1),h.amax(-1)],-1))
        raw_head = SignalHead(raw_seq_train.shape[1],k).to(dev)
    else: stats_head = None
    if raw_train is None: raw_head=None
    params=([] if a.raw_only else list(head.parameters())+[ce]+([] if q is None else [q]))+([] if stats_head is None else list(stats_head.parameters()))+([] if raw_head is None else list(raw_head.parameters())); opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-4)
    if a.task=='classification':
        weights=(1.0/torch.bincount(y).float())[y]; sampler=WeightedRandomSampler(weights,len(weights),replacement=True); loss_fn=nn.CrossEntropyLoss(); loader=DataLoader(TensorDataset(x,y,raw_train,raw_seq_train) if raw_train is not None else TensorDataset(x,y),batch_size=16,sampler=sampler)
    else: loader=DataLoader(TensorDataset(x,y_scaled,raw_train,raw_seq_train) if raw_train is not None else TensorDataset(x,y_scaled),batch_size=16,shuffle=True)
    for ep in range(a.epochs):
        for batch in loader:
            xb,yb=batch[:2]; xb,yb=xb.to(dev),yb.to(dev); z=xb + 0.01*torch.randn_like(xb) if head.training else xb; pred=head(z,ce,q)
            if a.task=='regression': pred=pred.reshape(-1)
            if a.raw_only: pred=torch.zeros_like(pred)
            if stats_head is not None: pred = pred + stats_head(batch[2].to(dev))
            if raw_head is not None: pred = pred + raw_head(batch[3].to(dev))
            loss=loss_fn(pred,yb.long() if a.task=='classification' else yb); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,2); opt.step()
    head.eval()
    with torch.no_grad():
        pred=head(xv.to(dev),ce,q)
        if a.task=='regression': pred=pred.reshape(-1)
        if a.raw_only: pred=torch.zeros_like(pred)
        if stats_head is not None: pred = pred + stats_head(raw_test.to(dev))
        if raw_head is not None: pred = pred + raw_head(raw_seq_test.to(dev))
        pred=pred.cpu()
    if a.task=='classification':
        metric={'balanced_accuracy':float(balanced_accuracy_score(yv,pred.argmax(-1))), 'macro_f1':float(f1_score(yv,pred.argmax(-1),average='macro'))}
    else:
        pred=torch.exp(pred*y_std+y_mean).squeeze(-1)
        metric={'mape':float(mean_absolute_percentage_error(yv.clamp_min(1e-6),pred.clamp_min(1e-6))), 'score':float(max(0,1-mean_absolute_percentage_error(yv.clamp_min(1e-6),pred.clamp_min(1e-6))))}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'head':head.state_dict(),'channel_embeddings':ce.detach().cpu(),'label_queries':None if q is None else q.detach().cpu(),'task':a.task,'metrics':metric},out.with_suffix('.pt')); json.dump({'dataset':ds.name,'task':a.task,'variant':a.variant,'n_train':len(y),'n_test':len(yv),'metrics':metric,'checkpoint':str(out.with_suffix('.pt'))},open(out,'w'),indent=2); print(json.dumps(metric),flush=True)
if __name__=='__main__': main()
