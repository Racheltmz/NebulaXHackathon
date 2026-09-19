import argparse,json,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score,f1_score
sys.path.insert(0,'src'); from train_ps3_classical import summary
ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--meta-c',type=float,default=1.0); a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)]); xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True); X=summary(x); Xt=summary(xt); cv=StratifiedKFold(5,shuffle=True,random_state=7)
bases=[make_pipeline(StandardScaler(),SVC(C=.5,gamma=2/X.shape[1],class_weight='balanced',probability=True,random_state=7)),make_pipeline(StandardScaler(),SVC(C=.2,gamma=2/X.shape[1],class_weight='balanced',probability=True,random_state=7)),make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')),make_pipeline(StandardScaler(),LogisticRegression(C=.2,class_weight='balanced',max_iter=3000))]
z=[];zt=[]
for b in bases:
 z.append(cross_val_predict(b,X,y,cv=cv,method='predict_proba')); b.fit(X,y); zt.append(b.predict_proba(Xt))
Z=np.concatenate(z,1); Zt=np.concatenate(zt,1); meta=LogisticRegression(C=a.meta_c,class_weight='balanced',max_iter=3000).fit(Z,y); p=meta.predict(Zt); metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,p)),'macro_f1':float(f1_score(yt,p,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump({'bases':bases,'meta':meta},out.with_suffix('.joblib')); json.dump({'model':'cv_stack','meta_c':a.meta_c,'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
