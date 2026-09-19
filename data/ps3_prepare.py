"""Convert the four PS3 raw schemas into model-ready [N,C,T] arrays."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def save(out, name, tx, ty, vx, vy):
    d = Path(out); d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, x, y in (("train", tx, ty), ("test", vx, vy)):
        xp, yp = d / f"{name}_{split}_x.npy", d / f"{name}_{split}_y.npy"
        if isinstance(x, list):
            arr = np.empty(len(x), dtype=object)
            for i, item in enumerate(x): arr[i] = item
        else:
            arr = np.asarray(x, dtype=np.float32)
        np.save(xp, arr, allow_pickle=True)
        np.save(yp, np.asarray(y), allow_pickle=True)
        paths[f"{split}_x"], paths[f"{split}_y"] = str(xp), str(yp)
    return paths

def split(xs, ys, seed=7):
    rng = np.random.default_rng(seed)
    # Stratify classification folds so rare faults are represented in the
    # held-out estimate; regression labels are effectively all unique.
    groups = {}
    for i, y in enumerate(ys): groups.setdefault(str(y), []).append(i)
    if len(groups) > len(ys) * .5:
        idx=np.arange(len(xs)); rng.shuffle(idx); cut=max(1,int(.8*len(idx)))
        return [xs[i] for i in idx[:cut]], [ys[i] for i in idx[:cut]], [xs[i] for i in idx[cut:]], [ys[i] for i in idx[cut:]]
    a=[]; b=[]
    for g in groups.values():
        g=np.asarray(g); rng.shuffle(g); cut=max(1, int(.8*len(g)))
        if len(g)>1: cut=min(cut,len(g)-1)
        a.extend(g[:cut]); b.extend(g[cut:])
    rng.shuffle(a); rng.shuffle(b)
    return [xs[i] for i in a], [ys[i] for i in a], [xs[i] for i in b], [ys[i] for i in b]

def door(root):
    base = Path(root) / "Door"; df = pd.read_csv(base / "Train.csv")
    ans = pd.read_csv(base / "Train_Segments_Answer.csv")
    def ts(s): return int("".join(f"{int(v):02d}" for v in str(s).split("-")))
    t = df.Datetime.map(ts).to_numpy(); cols = [c for c in df.columns if c != "Datetime"]
    xs, ys, groups = [], [], []
    for _, row in ans.iterrows():
        m = (t >= ts(row.start_time)) & (t <= ts(row.end_time)); z = df.loc[m, cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32).T
        if z.shape[-1] > 4: xs.append(z); ys.append(str(row.status))
    return split(xs, ys)

def rail(root):
    base = Path(root) / "Rail_Corrugation"; labels = pd.read_csv(base / "Train_Labels.csv").set_index("filename")
    xs, ys = [], []
    for f, y in labels.label.items():
        a = pd.read_csv(base / "Train" / f).to_numpy(np.float32); v = a[:, 1:].reshape(len(a), 8, 8, 2)
        # Validated compact Rail representation: speed and mean vibration /
        # shock for Side-I and Side-II.
        x = np.stack([a[:, 0], v[:, :, [0,2,4,6], 0].mean((1,2)),
                      v[:, :, [0,2,4,6], 1].mean((1,2)),
                      v[:, :, [1,3,5,7], 0].mean((1,2)),
                      v[:, :, [1,3,5,7], 1].mean((1,2))])
        xs.append(x); ys.append(str(y))
    return split(xs, ys)

def shm(root):
    base = Path(root) / "SHM"; labels = pd.read_csv(base / "Train_Labels.csv").set_index("filename")
    xs, ys = [], []
    for f, y in labels.damage.items():
        a = pd.read_csv(base / "Train" / f).select_dtypes(include="number").to_numpy(np.float32).T
        xs.append(a); ys.append(float(y))
    return split(xs, ys)

def acv(root):
    base = Path(root) / "ACV"; labels = pd.read_csv(base / "Train_Labels.csv").set_index("filename")
    files = sorted((base / "Train").glob("*.xlsx")); suffix_sets=[]; frames={}
    for f in files:
        d = pd.read_excel(f); frames[f.name]=d; suffix_sets.append({c.split(" - ",1)[1] for c in d.columns if c.startswith("Car ") and " - " in c})
    xs, ys, groups = [], [], []
    for gi, f in enumerate(files):
        d=frames[f.name]; faulty=str(labels.loc[f.name, "faulty_car"]).zfill(2)
        cars=sorted({c.split(" - ",1)[0].split()[-1] for c in d.columns if c.startswith("Car ") and " - " in c})
        case_x=[]; case_y=[]
        for car in cars:
            cols=[c for c in d.columns if c.startswith(f"Car {car} - ")]
            a=d[cols].apply(pd.to_numeric,errors="coerce").to_numpy(np.float32)
            a=np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # Cases expose different parameter subsets. Aggregate all
            # numeric signals within each car into stable channels rather
            # than intersecting suffix names across cases.
            a=np.stack([a.mean(1), a.std(1), np.quantile(a,.9,axis=1),
                        np.quantile(a,.1,axis=1)], axis=0)
            case_x.append(a); case_y.append(int(car == faulty))
        # Faults are defined relative to the other cars in the same train.
        # Add robust cross-car residual channels to make that invariant
        # explicit for both the frozen encoder and the trainable head.
        stack=np.stack(case_x)
        med=np.median(stack,axis=0); mad=np.median(np.abs(stack-med),axis=0)+1e-5
        for a,y in zip(case_x,case_y):
            xs.append(np.concatenate([a,(a-med)/mad],axis=0)); ys.append(y)
            groups.append(gi)
    # Hold out a complete train case: ACV is a within-train ranking problem,
    # and splitting individual cars would leak the case operating profile.
    case_ids=sorted(set(groups)); test_case=case_ids[-1]
    tr=[i for i,g in enumerate(groups) if g!=test_case]; te=[i for i,g in enumerate(groups) if g==test_case]
    return ([xs[i] for i in tr],[ys[i] for i in tr],[xs[i] for i in te],[ys[i] for i in te])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); out={}
    for name, fn in (("door",door),("rail",rail),("shm",shm),("acv",acv)):
        tx,ty,vx,vy=fn(a.root); out[name]=save(a.out,name,tx,ty,vx,vy); print(name,len(tx),len(vx),flush=True)
    json.dump(out,open(Path(a.out)/"manifest.json","w"),indent=2)
if __name__ == "__main__": main()
