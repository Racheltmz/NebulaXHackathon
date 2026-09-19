import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
sys.path.insert(0,'src')
from train_ps3_classical import summary

ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--depth',type=int,default=2); ap.add_argument('--estimators',type=int,default=300); a=ap.parse_args()
x=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_train_x.npy',allow_pickle=True)])
xt=np.stack([np.asarray(v,dtype=np.float32) for v in np.load(a.arrays+'_test_x.npy',allow_pickle=True)])
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True)
yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
X=summary(x); Xt=summary(xt); le=LabelEncoder(); yy=le.fit_transform(y); yyt=le.transform(yt)
counts=np.bincount(yy); w=len(yy)/(len(counts)*np.maximum(counts,1)); sw=w[yy]
m=XGBClassifier(n_estimators=a.estimators,max_depth=a.depth,learning_rate=.03,subsample=.85,colsample_bytree=.8,min_child_weight=2,reg_lambda=2,objective='multi:softprob',num_class=len(counts),eval_metric='mlogloss',tree_method='hist',random_state=7,n_jobs=4)
m.fit(X,yy,sample_weight=sw); p=le.inverse_transform(m.predict(Xt).astype(int))
metrics={'balanced_accuracy':float(balanced_accuracy_score(yt,p)),'macro_f1':float(f1_score(yt,p,average='macro'))}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(m,out.with_suffix('.joblib'))
json.dump({'model':'xgb','depth':a.depth,'estimators':a.estimators,'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
