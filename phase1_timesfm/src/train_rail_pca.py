import argparse,json
from pathlib import Path
import numpy as np,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,f1_score

ap=argparse.ArgumentParser()
ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True)
ap.add_argument('--components',type=int,default=20); ap.add_argument('--c',type=float,default=.5)
a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True)
yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
ix=np.linspace(0,x.shape[-1]-1,400).astype(int); x=x[:,:,ix]; xt=xt[:,:,ix]
xd=x[:,3:5].mean(1,keepdims=True)-x[:,1:3].mean(1,keepdims=True)
xtd=xt[:,3:5].mean(1,keepdims=True)-xt[:,1:3].mean(1,keepdims=True)
x=np.concatenate([x,xd],1).reshape(len(x),-1); xt=np.concatenate([xt,xtd],1).reshape(len(xt),-1)
m=make_pipeline(StandardScaler(),PCA(n_components=min(a.components,x.shape[0]-1),whiten=True,random_state=7),SVC(C=a.c,gamma='scale',class_weight='balanced'))
m.fit(x,y); p=m.predict(xt)
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,p)),'macro_f1':float(f1_score(yt,p,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(m,out.with_suffix('.joblib'))
json.dump({'model':'pca_rbf','components':a.components,'c':a.c,'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
