import argparse, json
from pathlib import Path
import numpy as np, torch, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from train_ps3_classical import summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--arrays',required=True); ap.add_argument('--features',required=True); ap.add_argument('--out',required=True); ap.add_argument('--c',type=float,default=.5); ap.add_argument('--gamma',default='scale'); a=ap.parse_args()
    x=np.load(a.arrays+'_train_x.npy',allow_pickle=True); xv=np.load(a.arrays+'_test_x.npy',allow_pickle=True); y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yv=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
    tr=torch.load(Path(a.features)/'train.pt',weights_only=False); te=torch.load(Path(a.features)/'test.pt',weights_only=False)
    def fv(z):
        q=torch.stack([v.float() for v in z]); return np.concatenate([q.mean((1,2)).numpy(),q.std((1,2),unbiased=False).numpy()],1)
    X=np.concatenate([summary(x),fv(tr['features'])],1); Xt=np.concatenate([summary(xv),fv(te['features'])],1)
    model=make_pipeline(StandardScaler(),SVC(C=a.c,gamma=a.gamma if a.gamma=='scale' else float(a.gamma)/X.shape[1],class_weight='balanced'))
    model.fit(X,y); pred=model.predict(Xt); metrics={'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'macro_f1':float(f1_score(yv,pred,average='macro')),'features':int(X.shape[1])}
    out=Path(a.out); joblib.dump(model,out.with_suffix('.joblib')); json.dump({'task':a.task,'c':a.c,'gamma':a.gamma,'n_train':len(y),'n_test':len(yv),'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
if __name__=='__main__': main()
