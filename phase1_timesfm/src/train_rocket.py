import argparse, json
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score
from sktime.classification.kernel_based import RocketClassifier

ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--kernels',type=int,default=2000); a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
# Standardize each sensor independently using training statistics; preserve shape.
mu=x.mean(axis=(0,2),keepdims=True); sd=x.std(axis=(0,2),keepdims=True); sd=np.maximum(sd,1e-5)
x=(x-mu)/sd; xt=(xt-mu)/sd
# Balance the highly skewed Rail training split explicitly because this
# sktime version does not expose class_weight on RocketClassifier.
counts={c:int((y==c).sum()) for c in np.unique(y)}; target=max(counts.values())
rng=np.random.default_rng(7); ix=np.concatenate([rng.choice(np.flatnonzero(y==c), target, replace=True) for c in counts])
xfit,yfit=x[ix],y[ix]
model=RocketClassifier(num_kernels=a.kernels, random_state=7, n_jobs=4)
model.fit(xfit,yfit); pred=model.predict(xt)
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
import joblib; joblib.dump(model,out.with_suffix('.joblib'))
json.dump({'model':'rocket','kernels':a.kernels,'n_train':len(y),'n_test':len(yt),'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
