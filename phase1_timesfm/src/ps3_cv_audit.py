import argparse,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score,f1_score,mean_absolute_percentage_error
from train_ps3_classical import summary

ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--task',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
base=Path(a.root)/a.task; x=np.load(str(base)+'_train_x.npy',allow_pickle=True); y=np.load(str(base)+'_train_y.npy',allow_pickle=True); X=summary(x,rich=False)
out={'task':a.task,'n_train':int(len(y)),'features':int(X.shape[1]),'folds':{}}
if a.task=='shm':
    yy=y.astype(float); z=np.log(np.maximum(yy,1e-5))
    for n in (3,5):
        kf=KFold(n,shuffle=True,random_state=7); pred=cross_val_predict(ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=7,n_jobs=4),X,z,cv=kf); pp=np.exp(pred); m=mean_absolute_percentage_error(yy,np.maximum(pp,1e-5)); out['folds'][str(n)]={'mape':float(m),'score':float(max(0,1-m))}
else:
    vals,counts=np.unique(y,return_counts=True); minc=int(counts.min())
    for n in (3,5):
        if minc<n: out['folds'][str(n)]={'skipped':f'min_class_count={minc}'}; continue
        sk=StratifiedKFold(n,shuffle=True,random_state=7); m=make_pipeline(StandardScaler(),SVC(C=.5,gamma=2.0/X.shape[1],class_weight='balanced')); pred=cross_val_predict(m,X,y,cv=sk); out['folds'][str(n)]={'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'class_counts':{str(k):int(v) for k,v in zip(vals,counts)}}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,'w'),indent=2); print(json.dumps(out),flush=True)
