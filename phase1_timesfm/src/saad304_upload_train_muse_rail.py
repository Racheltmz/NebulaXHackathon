import argparse,json
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import balanced_accuracy_score,f1_score
from sktime.classification.dictionary_based import MUSE

ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--window-inc',type=int,default=2); ap.add_argument('--anova',action='store_true'); a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True)
yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
ix=np.linspace(0,x.shape[-1]-1,500).astype(int); x=x[:,:,ix]; xt=xt[:,:,ix]
mu=x.mean((0,2),keepdims=True); sd=np.maximum(x.std((0,2),keepdims=True),1e-5); x=(x-mu)/sd; xt=(xt-mu)/sd
m=MUSE(anova=a.anova,variance=False,bigrams=True,window_inc=a.window_inc,alphabet_size=4,use_first_order_differences=True,feature_selection='chi2',n_jobs=4,random_state=7)
m.fit(x,y); p=m.predict(xt)
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,p)),'macro_f1':float(f1_score(yt,p,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(m,out.with_suffix('.joblib'))
json.dump({'model':'muse','window_inc':a.window_inc,'anova':a.anova,'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
