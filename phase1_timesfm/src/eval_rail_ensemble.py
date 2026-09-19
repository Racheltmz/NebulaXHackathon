import argparse, json
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import balanced_accuracy_score, f1_score
from train_ps3_classical import summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--arrays',required=True); ap.add_argument('--reports',required=True); ap.add_argument('--mode',default='mean'); ap.add_argument('--out',required=True); a=ap.parse_args()
    x=np.load(a.arrays+'_test_x.npy',allow_pickle=True); y=np.load(a.arrays+'_test_y.npy',allow_pickle=True); X=summary(x)
    names={'base':['ps3-rail-classical-svc05.joblib','ps3-rail-classical-svc10.joblib','ps3-rail-classical-svcg05.joblib'],
           'wide':['ps3-rail-classical-svc05.joblib','ps3-rail-classical-svc05g2.joblib','ps3-rail-classical-svcg05.joblib'],
           'smooth':['ps3-rail-classical-svc05.joblib','ps3-rail-classical-svc05g05.joblib','ps3-rail-classical-svc05g2.joblib'],
           'all':['ps3-rail-classical-svc05.joblib','ps3-rail-classical-svc10.joblib','ps3-rail-classical-svcg05.joblib','ps3-rail-classical-svc05g2.joblib']}[a.mode]
    scores=[]
    for n in names:
        m=joblib.load(Path(a.reports)/n); s=m.decision_function(X); s=(s-s.mean(1,keepdims=True))/(s.std(1,keepdims=True)+1e-6); scores.append(s)
    pred=np.stack(scores).mean(0).argmax(1); pred=joblib.load(Path(a.reports)/names[0]).classes_[pred]
    metrics={'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'models':names}
    json.dump(metrics,open(a.out,'w'),indent=2); print(json.dumps(metrics),flush=True)
if __name__=='__main__': main()
