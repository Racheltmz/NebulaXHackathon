import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
sys.path.insert(0,'src')
from train_ps3_classical import summary

ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--scale',type=float,default=.5); a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
X=summary(x); Xt=summary(xt); classes=np.unique(y)
def make_model():
    return make_pipeline(StandardScaler(),SVC(C=.5,gamma=2.0/X.shape[1],class_weight='balanced',decision_function_shape='ovr'))
cv=StratifiedKFold(5,shuffle=True,random_state=7)
scores=cross_val_predict(make_model(),X,y,cv=cv,method='decision_function')
best=(-1,None); grid=np.arange(-a.scale,a.scale+.001,a.scale/5)
for b1 in grid:
    for b2 in grid:
        bias=np.array([0,b1,b2],dtype=np.float32); pred=classes[(scores+bias[None,:]).argmax(1)]; val=balanced_accuracy_score(y,pred)
        if val>best[0]: best=(val,bias.copy())
m=make_model(); m.fit(X,y); pred=classes[(m.decision_function(Xt)+best[1][None,:]).argmax(1)]
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro')),'cv_balanced_accuracy':float(best[0]),'bias':best[1].tolist()}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump({'model':m,'bias':best[1],'classes':classes},out.with_suffix('.joblib'))
json.dump({'model':'svc05g2_cv_calibrated','scale':a.scale,'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
