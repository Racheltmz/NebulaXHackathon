import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
import joblib
ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--neighbors',type=int,default=3); ap.add_argument('--distance',default='dtw'); a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
# Downsample long Rail traces while preserving all five sensor channels.
ix=np.linspace(0,x.shape[-1]-1,500).astype(int); x=x[:,:,ix]; xt=xt[:,:,ix]
mu=x.mean(axis=(0,2),keepdims=True); sd=np.maximum(x.std(axis=(0,2),keepdims=True),1e-5); x=(x-mu)/sd; xt=(xt-mu)/sd
m=KNeighborsTimeSeriesClassifier(n_neighbors=a.neighbors,distance=a.distance,n_jobs=4,weights='distance')
m.fit(x,y); p=m.predict(xt); metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,p)),'macro_f1':float(f1_score(yt,p,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(m,out.with_suffix('.joblib')); json.dump({'model':'knn_ts','neighbors':a.neighbors,'distance':a.distance,'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
