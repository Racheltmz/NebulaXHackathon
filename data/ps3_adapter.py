"""PS3-facing data adapter: raw per-window sensor arrays -> model tensors.

The hackathon's four subsystems use different file schemas. This adapter keeps
the classifier interface uniform: [examples, channels, time] plus labels, while
preserving IDs for prediction CSV generation. It accepts npy/npz/csv/parquet and
can be configured per subsystem once the Info Kits are inspected on the cluster.
"""
from pathlib import Path
import numpy as np

def load_array(path):
    path = Path(path)
    if path.suffix == ".npy": return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True); return z[z.files[0]]
    if path.suffix == ".parquet":
        import pandas as pd; return pd.read_parquet(path)
    if path.suffix in (".csv", ".txt"):
        import pandas as pd; return pd.read_csv(path)
    raise ValueError(f"Unsupported PS3 file: {path}")

def windows_from_frame(frame, time_columns=None, group_columns=None, label_column=None):
    """Convert a long sensor table into padded [N,C,T] windows.

    group_columns define one independent sensor window. If absent, the complete
    table is treated as one window. Non-numeric columns are metadata and are
    excluded from the signal unless explicitly listed in time_columns.
    """
    import pandas as pd
    if not isinstance(frame, pd.DataFrame): frame = pd.DataFrame(frame)
    group_columns = group_columns or []
    if time_columns is None:
        excluded = set(group_columns + ([label_column] if label_column else []))
        time_columns = [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
    groups = frame.groupby(group_columns, sort=False) if group_columns else [(None, frame)]
    xs, ys, ids = [], [], []
    for key, g in groups:
        arr = g[time_columns].to_numpy(dtype=np.float32).T
        xs.append(arr); ids.append(key)
        if label_column: ys.append(g[label_column].iloc[0])
    if ys: return xs, np.asarray(ys), ids
    return xs, None, ids

def resample_or_pad(x, length):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] == length: return x
    if x.shape[-1] > length: return x[..., -length:]
    return np.pad(x, [(0, 0), (length - x.shape[-1], 0)])
