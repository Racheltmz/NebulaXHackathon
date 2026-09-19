import argparse,json
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from train_ps3_classical import summary
ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
x=np.load(a.arrays+'_train_x.npy',allow_pickle=True); xt=np.load(a.arrays+'_test_x.npy',allow_pickle=True)
y=np.load(a.arrays+'_train_y.npy',allow_pickle=True).astype(float); yt=np.load(a.arrays+'_test_y.npy',allow_pickle=True).astype(float)
X=summary(x,rich=True); Xt=summary(xt,rich=True)
if a.model=='extra': m=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=1,max_features=.8,random_state=7,n_jobs=4)
elif a.model=='extra2': m=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=2,max_features=1.0,random_state=7,n_jobs=4)
elif a.model=='rf': m=RandomForestRegressor(n_estimators=800,min_samples_leaf=1,max_features=.8,random_state=7,n_jobs=4)
else: m=HistGradientBoostingRegressor(max_iter=500,max_leaf_nodes=15,l2_regularization=1e-2,random_state=7)
m.fit(X,np.log(np.maximum(y,1e-5))); p=np.exp(m.predict(Xt)); mape=mean_absolute_percentage_error(yt,np.maximum(p,1e-5)); met={'mape':float(mape),'score':float(max(0,1-mape)),'model':a.model,'features':int(X.shape[1])}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(m,out.with_suffix('.joblib')); json.dump({'metrics':met,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(met),flush=True)
