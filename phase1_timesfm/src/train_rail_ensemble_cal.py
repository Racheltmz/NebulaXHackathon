import argparse,json,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,f1_score
sys.path.insert(0,'src'); from train_ps3_classical import summary
ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--rich',action='store_true'); ap.add_argument('--seed',type=int,default=19); a=ap.parse_args()
x=np.load(a.arrays+'_train_x.npy',allow_pickle=True); xt=np.load(a.arrays+'_test_x.npy',allow_pickle=True); y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
X=summary(x,rich=a.rich); Xt=summary(xt,rich=a.rich); classes=np.unique(y); k=X.shape[1]
spec=[(.2,.5),( .5,.5),(1.,.5),(.5,2.),(1.,2.)]
cv=StratifiedKFold(5,shuffle=True,random_state=a.seed); oofs=[]; models=[]
for c,g in spec:
 m=make_pipeline(StandardScaler(),SVC(C=c,gamma=g/k,class_weight='balanced',decision_function_shape='ovr'))
 oofs.append(cross_val_predict(m,X,y,cv=cv,method='decision_function')); m.fit(X,y); models.append(m)
oofs=np.stack(oofs); rng=np.random.default_rng(a.seed); best=(-1,None)
for _ in range(4000):
 w=rng.dirichlet(np.ones(len(spec))); bias=rng.uniform(-.4,.4,len(classes)); score=(oofs*w[:,None,None]).sum(0)+bias; pred=classes[score.argmax(1)]; v=balanced_accuracy_score(y,pred)
 if v>best[0]: best=(v,(w,bias))
w,bias=best[1]; test=sum(w[i]*models[i].decision_function(Xt) for i in range(len(models)))+bias; pred=classes[test.argmax(1)]
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro')),'cv_balanced_accuracy':float(best[0]),'weights':w.tolist(),'bias':bias.tolist(),'rich':a.rich}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump({'models':models,'weights':w,'bias':bias,'classes':classes},out.with_suffix('.joblib')); json.dump({'model':'svc_ensemble_cv','metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
