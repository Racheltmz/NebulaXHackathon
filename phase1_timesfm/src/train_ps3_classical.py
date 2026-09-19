import argparse, json, os
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_percentage_error

def augment_minority(x, y, mode, copies=2):
    if not mode: return x, y
    rng=np.random.default_rng(17); classes, counts=np.unique(y, return_counts=True); target=int(counts.max())
    rows=[np.asarray(v,dtype=np.float32) for v in x]; labels=[v for v in y]
    for c,n in zip(classes,counts):
        if n==target: continue
        base=np.flatnonzero(y==c)
        for j in range(int(copies)):
            for idx in base:
                z=np.asarray(x[idx],dtype=np.float32).copy()
                if 'noise' in mode:
                    z += rng.normal(0, 0.01*np.maximum(z.std(axis=-1,keepdims=True),1e-5), z.shape).astype(np.float32)
                if 'roll' in mode: z=np.roll(z, int(rng.choice([-200,-100,100,200])), axis=-1)
                if 'scale' in mode: z *= np.float32(rng.uniform(.98,1.02))
                rows.append(z); labels.append(c)
    return np.asarray(rows,dtype=object), np.asarray(labels)

def summary(arr, full_channels=False, rich=False, sidewave=False):
    rows=[]
    for z in arr:
        z=np.asarray(z,dtype=np.float32)
        if z.ndim == 0: z=z.reshape(1,1)
        if z.ndim==1: z=z[None,:]
        original=z.shape[-1]
        if z.shape[0] >= 17 and not full_channels:
            # The raw Rail adapter stores four sensor statistics per
            # side/measurement. Collapse those statistics for the compact
            # low-sample SVM while retaining the richer array for neural
            # heads that can model it explicitly.
            z=np.stack([z[0],z[1:5].mean(0),z[5:9].mean(0),z[9:13].mean(0),z[13:17].mean(0)])
        if original>4096: z=z[:,np.linspace(0,original-1,4096).astype(int)]
        d=np.diff(z,axis=-1); q=np.quantile(z,[.05,.1,.25,.5,.75,.9,.95],axis=-1).reshape(-1)
        mean,std=z.mean(-1),z.std(-1); rms=np.sqrt((z*z).mean(-1)); peak=np.abs(z).max(-1)
        sp=np.abs(np.fft.rfft(z,axis=-1))**2; n=sp.shape[-1]
        bands=np.stack([sp[:,:max(2,n//16)].mean(-1),sp[:,n//16:max(2,n//8)].mean(-1),sp[:,n//8:max(3,n//4)].mean(-1),sp[:,n//4:].mean(-1)],1).reshape(-1)
        extra=np.empty(0,dtype=np.float32)
        if z.shape[0]>=17 and not full_channels:
            # Expanded Rail layout: speed, then four statistics for each of
            # Side-I vibration/shock and Side-II vibration/shock.
            gm=np.array([mean[1:5].mean(),mean[5:9].mean(),mean[9:13].mean(),mean[13:17].mean()])
            gs=np.array([std[1:5].mean(),std[5:9].mean(),std[9:13].mean(),std[13:17].mean()])
            gr=np.array([rms[1:5].mean(),rms[5:9].mean(),rms[9:13].mean(),rms[13:17].mean()])
            gp=np.array([peak[1:5].mean(),peak[5:9].mean(),peak[9:13].mean(),peak[13:17].mean()])
            extra=np.concatenate([gm[2:]-gm[:2],gs[2:]-gs[:2],gr[2:]-gr[:2],gp[2:]-gp[:2]])
        elif z.shape[0]>=5:
            si=z[1:3].mean(0); sj=z[3:5].mean(0)
            side_q=np.quantile(sj,[.05,.1,.25,.5,.75,.9,.95])-np.quantile(si,[.05,.1,.25,.5,.75,.9,.95])
            side_d=sj-si
            extra=np.concatenate([mean[3:5]-mean[1:3],std[3:5]-std[1:3],rms[3:5]-rms[1:3],peak[3:5]-peak[1:3],side_q,[side_d.std(),side_d.mean(),np.abs(side_d).max()]])
        feat=np.concatenate([mean,std,rms,peak,q,d.std(-1),d.mean(-1),bands, np.array([np.log1p(original)],np.float32),extra])
        if rich:
            # Cross-sensor structure: correlations and robust derivative
            # quantiles expose side asymmetry not represented by marginals.
            if z.shape[0] > 1:
                corr=np.corrcoef(z); corr=np.nan_to_num(corr,nan=0.0)[np.triu_indices(z.shape[0],1)]
            else:
                corr=np.empty(0,dtype=np.float32)
            dq=np.quantile(d,[.1,.5,.9],axis=-1).reshape(-1)
            feat=np.concatenate([feat,corr.astype(np.float32),dq.astype(np.float32)])
        if sidewave and z.shape[0]>=5:
            sdif=z[3:5].mean(0)-z[1:3].mean(0); ss=z[3:5].mean(0)+z[1:3].mean(0)
            def wf(v):
                vv=np.diff(v); spv=np.abs(np.fft.rfft(v))**2; nn=spv.shape[0]
                return np.concatenate([np.array([v.mean(),v.std(),np.sqrt((v*v).mean()),np.abs(v).max()],np.float32),
                    np.quantile(v,[.01,.05,.1,.25,.5,.75,.9,.95,.99]).astype(np.float32),
                    np.array([vv.std(),vv.mean(),(np.signbit(vv[1:])!=np.signbit(vv[:-1])).mean()],np.float32),
                    np.array([spv[:max(2,nn//16)].mean(),spv[nn//16:max(2,nn//8)].mean(),spv[nn//8:max(3,nn//4)].mean(),spv[nn//4:].mean(),(spv*np.arange(nn)).sum()/(spv.sum()+1e-6)],np.float32)])
            feat=np.concatenate([feat,wf(sdif),wf(ss),wf(np.abs(sdif))])
        rows.append(feat)
    return np.nan_to_num(np.stack(rows).astype(np.float32))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--arrays',required=True); ap.add_argument('--out',required=True); ap.add_argument('--model',default='svc'); ap.add_argument('--full-channels',action='store_true'); ap.add_argument('--rich',action='store_true'); ap.add_argument('--sidewave',action='store_true'); ap.add_argument('--augment',default=''); ap.add_argument('--augment-copies',type=int,default=2); a=ap.parse_args()
    x=np.load(a.arrays+'_train_x.npy',allow_pickle=True); xv=np.load(a.arrays+'_test_x.npy',allow_pickle=True)
    y=np.load(a.arrays+'_train_y.npy',allow_pickle=True); yv=np.load(a.arrays+'_test_y.npy',allow_pickle=True)
    x,y=augment_minority(x,y,a.augment,a.augment_copies)
    X=summary(x, a.full_channels, a.rich, a.sidewave); Xt=summary(xv, a.full_channels, a.rich, a.sidewave)
    if a.task=='regression':
        model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=7,n_jobs=4)
        model.fit(X,np.log(np.maximum(y.astype(float),1e-5))); pred=np.exp(model.predict(Xt)); m=mean_absolute_percentage_error(yv.astype(float),np.maximum(pred,1e-5)); metrics={'mape':float(m),'score':float(max(0,1-m))}
    else:
        if a.model.startswith('side_sel'):
            kk=int(a.model.replace('side_sel','')); model=make_pipeline(StandardScaler(),SelectKBest(f_classif,k=min(kk,X.shape[1])),SVC(C=.5,gamma='scale',class_weight='balanced'))
        elif a.model=='extra': model=ExtraTreesClassifier(n_estimators=500,class_weight='balanced',max_features=.8,random_state=7,n_jobs=4,min_samples_leaf=1)
        elif a.model=='rf': model=RandomForestClassifier(n_estimators=500,class_weight='balanced_subsample',random_state=7,n_jobs=4,min_samples_leaf=1)
        elif a.model=='svc01': model=make_pipeline(StandardScaler(),SVC(C=.1,gamma='scale',class_weight='balanced'))
        elif a.model=='svc05': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma='scale',class_weight='balanced'))
        elif a.model=='svc10': model=make_pipeline(StandardScaler(),SVC(C=1.0,gamma='scale',class_weight='balanced'))
        elif a.model=='svc20': model=make_pipeline(StandardScaler(),SVC(C=2.0,gamma='scale',class_weight='balanced'))
        elif a.model=='svc05g2': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc02g2': model=make_pipeline(StandardScaler(),SVC(C=.2,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc03g2': model=make_pipeline(StandardScaler(),SVC(C=.3,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc04g2': model=make_pipeline(StandardScaler(),SVC(C=.4,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc06g2': model=make_pipeline(StandardScaler(),SVC(C=.6,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc08g2': model=make_pipeline(StandardScaler(),SVC(C=.8,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc07g2': model=make_pipeline(StandardScaler(),SVC(C=.7,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc1g2': model=make_pipeline(StandardScaler(),SVC(C=1.0,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc10g2': model=make_pipeline(StandardScaler(),SVC(C=1.0,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc05g05': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma=.5/ X.shape[1],class_weight='balanced'))
        elif a.model=='svc20': model=make_pipeline(StandardScaler(),SVC(C=2.0,gamma='scale',class_weight='balanced'))
        elif a.model=='svc50': model=make_pipeline(StandardScaler(),SVC(C=5.0,gamma='scale',class_weight='balanced'))
        elif a.model=='svcg05': model=make_pipeline(StandardScaler(),SVC(C=1.0,gamma=.5/ X.shape[1],class_weight='balanced'))
        elif a.model=='svcg2': model=make_pipeline(StandardScaler(),SVC(C=1.0,gamma=2.0/ X.shape[1],class_weight='balanced'))
        elif a.model=='lda': model=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
        elif a.model=='knn3': model=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3,weights='distance'))
        elif a.model=='knn5': model=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5,weights='distance'))
        elif a.model=='logit': model=make_pipeline(StandardScaler(),LogisticRegression(C=.2,class_weight='balanced',max_iter=3000))
        elif a.model=='sel8': model=make_pipeline(StandardScaler(),SelectKBest(f_classif,k=min(8,X.shape[1])),SVC(C=.5,gamma='scale',class_weight='balanced'))
        elif a.model=='sel16': model=make_pipeline(StandardScaler(),SelectKBest(f_classif,k=min(16,X.shape[1])),SVC(C=.5,gamma='scale',class_weight='balanced'))
        elif a.model=='sel32': model=make_pipeline(StandardScaler(),SelectKBest(f_classif,k=min(32,X.shape[1])),SVC(C=.5,gamma='scale',class_weight='balanced'))
        elif a.model=='sel64': model=make_pipeline(StandardScaler(),SelectKBest(f_classif,k=min(64,X.shape[1])),SVC(C=.5,gamma='scale',class_weight='balanced'))
        elif a.model=='w122': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma='scale',class_weight={'Normal':1.0,'Side I':2.0,'Side II':2.0}))
        elif a.model=='w132': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma='scale',class_weight={'Normal':1.0,'Side I':3.0,'Side II':2.0}))
        elif a.model=='w123': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma='scale',class_weight={'Normal':1.0,'Side I':2.0,'Side II':3.0}))
        elif a.model=='w022': model=make_pipeline(StandardScaler(),SVC(C=.5,gamma='scale',class_weight={'Normal':.7,'Side I':2.0,'Side II':2.0}))
        else: model=make_pipeline(StandardScaler(),SVC(C=2.0,gamma='scale',class_weight='balanced'))
        model.fit(X,y); pred=model.predict(Xt); metrics={'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'macro_f1':float(f1_score(yv,pred,average='macro'))}
        if 'acv' in str(a.arrays).lower() and hasattr(model, 'decision_function'):
            score=model.decision_function(Xt)
            if np.ndim(score)>1: score=score[:, -1]
            order=np.argsort(-score); ranks=np.flatnonzero(np.asarray(yv)[order]==1)
            if len(ranks):
                rank=int(ranks[0])+1; metrics.update({'fault_rank':rank,'rank1':float(rank==1),'reciprocal_rank':float(1.0/rank)})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    import joblib; joblib.dump(model,out.with_suffix('.joblib'))
    json.dump({'task':a.task,'model':a.model,'n_train':len(y),'n_test':len(yv),'metrics':metrics,'checkpoint':str(out.with_suffix('.joblib'))},open(out,'w'),indent=2); print(json.dumps(metrics),flush=True)
if __name__=='__main__': main()
