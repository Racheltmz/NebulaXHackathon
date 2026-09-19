"""Per-recording feature extractors for the `classical_nx` Rail/SHM models.

Copied from the independent `nebulax` (ian-classical) work -- the same code that
produced the cached feature tables the nx models were trained on -- with only the
file loading replaced by array/DataFrame inputs so it can run on unseen test
files.  Source scripts:
  rail : src/ps3_eda_and_probe.py (rail_features), rail/scripts/ps3_rail_features_v3.py (extra),
         rail/scripts/ps3_rail_relative_features.py (relative), rail/scripts/ps3_rail_phase_features.py
  shm  : src/ps3_shm_features.py, ps3_shm_multiscale_probe.py, ps3_shm_rainflow_features.py,
         ps3_shm_temporal_features.py
All transforms are label-free and per-recording; column ORDER is fixed by the cached
CSV headers (see final_data.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

# ------------------------------------------------------------------ Rail ----


def rail_v1(df: pd.DataFrame) -> dict:
    x = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    vib = x[:, 0::2].reshape(len(x), 8, 8)
    shock = x[:, 1::2].reshape(len(x), 8, 8)
    out = {"n_rows": float(len(x)), "speed_mean": float(df.iloc[:, 0].mean())}
    for side, pos in (("side1", [0, 2, 4, 6]), ("side2", [1, 3, 5, 7])):
        for kind, a in (("vib", vib[:, :, pos]), ("shock", shock[:, :, pos])):
            z = a.reshape(len(x), -1)
            rms = np.sqrt(np.mean(z * z, axis=0))
            absz = np.abs(z)
            out[f"{side}_{kind}_rms_mean"] = float(rms.mean())
            out[f"{side}_{kind}_rms_max"] = float(rms.max())
            out[f"{side}_{kind}_p99_mean"] = float(np.quantile(absz, .99, axis=0).mean())
            out[f"{side}_{kind}_crest_mean"] = float((absz.max(0) / (rms + 1e-9)).mean())
            out[f"{side}_{kind}_kurt_mean"] = float(np.mean(kurtosis(z, axis=0, fisher=False, bias=False)))
            out[f"{side}_{kind}_skew_abs_mean"] = float(np.mean(np.abs(skew(z, axis=0, bias=False))))
            fft = np.abs(np.fft.rfft(z - z.mean(0), axis=0)) ** 2
            total = fft[1:].sum(0) + 1e-9
            out[f"{side}_{kind}_spec_peak_ratio"] = float(np.max(fft[1:], axis=0).mean() / total.mean())
            out[f"{side}_{kind}_high_ratio"] = float(fft[int(len(fft) * .25):].sum() / fft[1:].sum())
    for suffix in ("vib_rms_mean", "vib_rms_max", "vib_p99_mean", "vib_crest_mean", "vib_kurt_mean",
                   "vib_spec_peak_ratio", "vib_high_ratio", "shock_rms_mean", "shock_rms_max", "shock_p99_mean",
                   "shock_crest_mean", "shock_kurt_mean", "shock_spec_peak_ratio", "shock_high_ratio"):
        out[f"diff_{suffix}"] = out[f"side1_{suffix}"] - out[f"side2_{suffix}"]
    return out


def rail_extra(df: pd.DataFrame) -> dict:
    x = df.iloc[:, 1:].to_numpy(float)
    vib = x[:, 0::2].reshape(len(x), 8, 8)
    shock = x[:, 1::2].reshape(len(x), 8, 8)
    o = {}
    for side, ps in (("side1", [0, 2, 4, 6]), ("side2", [1, 3, 5, 7])):
        for kind, a in (("vib", vib), ("shock", shock)):
            z = a[:, :, ps].reshape(len(x), -1)
            z = z - z.mean(0)
            p = np.abs(np.fft.rfft(z, axis=0)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / 10000)
            p[0] = 0
            total = p.sum(0) + 1e-9
            ix = p.argmax(0)
            o[f"{side}_{kind}_peak_hz_mean"] = float(np.average(f[ix], weights=np.maximum(p.max(0), 1e-9)))
            o[f"{side}_{kind}_peak_hz_median"] = float(np.median(f[ix]))
            o[f"{side}_{kind}_peak_ratio_median"] = float(np.median(p.max(0) / total))
            for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 5000)):
                q = p[(f >= lo) & (f < hi)].sum() / total.sum()
                o[f"{side}_{kind}_band_{lo}_{hi}"] = float(q)
    for k in list(o):
        if k.startswith("side1_") and k.replace("side1_", "side2_") in o:
            o["diff_" + k[6:]] = o[k] - o[k.replace("side1_", "side2_")]
    return o


def rail_v3(df: pd.DataFrame) -> dict:
    r = rail_v1(df)
    r.update(rail_extra(df))
    return r


def rail_relative(v3: dict) -> dict:
    """Scale-normalised Side-I/Side-II contrasts (mirrors ps3_rail_relative_features.py)."""
    base = {k: v for k, v in v3.items() if k != "n_rows"}
    out = dict(base)
    for c in list(base):
        if c.startswith("side1_"):
            other = "side2_" + c[len("side1_"):]
            if other in base:
                a, b = float(base[c]), float(base[other])
                scale = abs(a) + abs(b) + 1e-8
                q = c[len("side1_"):]
                out[f"rel_diff_{q}"] = (a - b) / scale
                out[f"log_ratio_{q}"] = np.sign(a) * np.log1p(abs(a)) - np.sign(b) * np.log1p(abs(b))
                out[f"side_sum_{q}"] = a + b
                out[f"side_max_{q}"] = max(a, b)
    return out


NBINS = 64


def rail_phase(df: pd.DataFrame) -> dict:
    t = df.iloc[:, 0].to_numpy(float)
    x = df.iloc[:, 1:].to_numpy(float)
    n = len(x)
    edges = np.flatnonzero(np.diff(t) != 0) + 1
    if len(edges) >= 4:
        left, right = int(edges[0]), int(edges[-1])
        anchors = np.r_[left, edges[1:-1], right]
        phase = np.interp(np.arange(left, right), anchors, np.arange(len(anchors), dtype=float))
        phase = (phase - np.floor(phase))
        xx = x[left:right]
    else:
        phase = np.linspace(0, 1, n, endpoint=False)
        xx = x
    bi = np.minimum((phase * NBINS).astype(int), NBINS - 1)
    vib = xx[:, 0::2].reshape(len(xx), 8, 8)
    shock = xx[:, 1::2].reshape(len(xx), 8, 8)
    out = {'tacho_transitions': float(len(edges)),
           'tacho_speed_mps': float(max(len(edges) - 1, 0) / (2 * 90) * .85)}
    profiles = {}
    for side, pos in (('s1', [0, 2, 4, 6]), ('s2', [1, 3, 5, 7])):
        for kind, arr in (('v', vib), ('h', shock)):
            z = arr[:, :, pos].reshape(len(xx), -1)
            prof = np.zeros((NBINS, z.shape[1]))
            cnt = np.bincount(bi, minlength=NBINS).astype(float)
            for j in range(z.shape[1]):
                prof[:, j] = np.bincount(bi, weights=np.abs(z[:, j]), minlength=NBINS)
            prof /= np.maximum(cnt[:, None], 1)
            profiles[(side, kind)] = prof.mean(1)
            for stat, a in (('mean', prof.mean(1)), ('std', prof.std(1)),
                            ('peak', prof.max(1)), ('q90', np.quantile(prof, .9, axis=1))):
                out[f'{side}_{kind}_{stat}_mean'] = float(a.mean())
                out[f'{side}_{kind}_{stat}_std'] = float(a.std())
                out[f'{side}_{kind}_{stat}_max'] = float(a.max())
    for kind in ('v', 'h'):
        d = profiles[('s1', kind)] - profiles[('s2', kind)]
        s = profiles[('s1', kind)] + profiles[('s2', kind)] + 1e-9
        for h in (1, 2, 3, 4, 8):
            c = np.cos(2 * np.pi * h * np.arange(NBINS) / NBINS)
            q = np.sin(2 * np.pi * h * np.arange(NBINS) / NBINS)
            out[f'd_{kind}_cos{h}'] = float(np.dot(d, c) / NBINS)
            out[f'd_{kind}_sin{h}'] = float(np.dot(d, q) / NBINS)
            out[f'r_{kind}_cos{h}'] = float(np.dot(d / s, c) / NBINS)
            out[f'r_{kind}_sin{h}'] = float(np.dot(d / s, q) / NBINS)
        out[f'd_{kind}_abs_mean'] = float(np.mean(np.abs(d)))
        out[f'd_{kind}_abs_p90'] = float(np.quantile(np.abs(d), .9))
        out[f'd_{kind}_signed_std'] = float(np.std(d))
    return out


# ------------------------------------------------------------------- SHM ----


def shm_generic(x: np.ndarray) -> dict:
    x = x.astype("float32")
    a = np.abs(x)
    d = np.diff(x)
    q = np.quantile(x, [.01, .05, .1, .25, .5, .75, .9, .95, .99])
    aq = np.quantile(a, [.5, .9, .95, .99, .999])
    out = {"mean": x.mean(), "std": x.std(), "rms": np.sqrt(np.mean(x * x)), "abs_mean": a.mean(), "abs_std": a.std(),
           "min": x.min(), "max": x.max(), "ptp": np.ptp(x), "skew": skew(x), "kurtosis": kurtosis(x, fisher=False),
           "diff_std": d.std(), "diff_abs_mean": np.abs(d).mean(),
           "zero_cross": np.mean((x[:-1] - x.mean()) * (x[1:] - x.mean()) < 0), "n": len(x)}
    out.update({f"q{i}": v for i, v in enumerate(q)})
    out.update({f"absq{i}": v for i, v in enumerate(aq)})
    y = x[::16]
    peaks, _ = find_peaks(y, distance=2)
    troughs, _ = find_peaks(-y, distance=2)
    tp = np.sort(np.r_[peaks, troughs])
    ranges = np.abs(np.diff(y[tp])) if len(tp) > 1 else np.array([0.])
    out.update({"turning_points": len(tp), "range_mean": ranges.mean(), "range_std": ranges.std(),
                "range_p90": np.quantile(ranges, .9), "range_p99": np.quantile(ranges, .99), "range_max": ranges.max()})
    n = (len(x) // 64) * 64
    z = x[:n].reshape(-1, 64).mean(1)
    power = np.abs(np.fft.rfft(z - z.mean())) ** 2
    power[0] = 0
    freq = np.fft.rfftfreq(len(z), 64 / 10000)
    total = power.sum() + 1e-9
    out.update({"spec_peak_hz": freq[power.argmax()],
                "spec_entropy": float(-np.sum((power / total) * np.log(power / total + 1e-12)))})
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 500), (500, 5000)):
        out[f"band_{lo}_{hi}"] = power[(freq >= lo) & (freq < hi)].sum() / total
    return out


def shm_multiscale(x: np.ndarray) -> dict:
    x = x.astype("float32")
    out = {"n": float(len(x))}
    for nw in (8, 16, 32, 64, 128):
        z = x[:(len(x) // nw) * nw].reshape(nw, -1)
        t = np.linspace(0, 1, nw)
        vals = {"mean": z.mean(1), "std": z.std(1), "rms": np.sqrt(np.mean(z * z, 1)), "ptp": np.ptp(z, 1),
                "absmean": np.abs(z).mean(1), "maxabs": np.abs(z).max(1), "diffstd": np.diff(z, axis=1).std(1)}
        for name, v in vals.items():
            q = np.quantile(v, [.05, .1, .25, .5, .75, .9, .95, .99])
            for i, a in enumerate(q):
                out[f"w{nw}_{name}_q{i}"] = float(a)
            out[f"w{nw}_{name}_first"] = float(v[0])
            out[f"w{nw}_{name}_last"] = float(v[-1])
            out[f"w{nw}_{name}_delta"] = float(v[-1] - v[0])
            out[f"w{nw}_{name}_stdtime"] = float(v.std())
            out[f"w{nw}_{name}_max"] = float(v.max())
            out[f"w{nw}_{name}_min"] = float(v.min())
            out[f"w{nw}_{name}_slope"] = float(np.polyfit(t, v, 1)[0])
            out[f"w{nw}_{name}_auc"] = float(np.trapezoid(v, t) if hasattr(np, "trapezoid") else np.trapz(v, t))
    return out


def _reversals(x):
    peaks, _ = find_peaks(x)
    troughs, _ = find_peaks(-x)
    ix = np.sort(np.r_[0, peaks, troughs, len(x) - 1])
    y = x[ix]
    keep = np.r_[True, np.diff(y) != 0]
    return y[keep]


def _cycles(rev):
    stack = []
    out = []
    for v in rev:
        stack.append(float(v))
        while len(stack) >= 3:
            r1 = abs(stack[-2] - stack[-3])
            r2 = abs(stack[-1] - stack[-2])
            if r1 < r2:
                break
            if len(stack) == 3:
                out.append((r1, .5))
                del stack[0]
            else:
                out.append((r1, 1.0))
                del stack[-3:-1]
    out.extend((abs(stack[i + 1] - stack[i]), .5) for i in range(len(stack) - 1))
    if not out:
        return np.zeros(1), np.ones(1)
    r, c = np.asarray(out, dtype="float64").T
    return r, c


def shm_rainflow(x: np.ndarray) -> dict:
    x = x.astype("float32")
    out = {"n": float(len(x)), "std": float(x.std()), "ptp": float(np.ptp(x))}
    for ds in (4, 16, 64, 256):
        r, c = _cycles(_reversals(x[::ds]))
        out[f"rf{ds}_count"] = float(c.sum())
        out[f"rf{ds}_reversals"] = float(len(r))
        for q, v in zip((.5, .75, .9, .95, .99, .999, 1.0), np.quantile(r, [.5, .75, .9, .95, .99, .999, 1.0])):
            out[f"rf{ds}_q{q:g}"] = float(v)
        for m in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            out[f"rf{ds}_damage{m}"] = float(np.sum(c * np.maximum(r, 1e-12) ** m))
        out[f"rf{ds}_range_mean"] = float(np.average(r, weights=c))
        out[f"rf{ds}_range_std"] = float(np.sqrt(np.average((r - np.average(r, weights=c)) ** 2, weights=c)))
    return out


def shm_temporal(x: np.ndarray, windows: int = 128) -> dict:
    x = x.astype("float32")
    n = (len(x) // windows) * windows
    z = x[:n].reshape(windows, -1)
    t = np.linspace(0, 1, windows)
    absz = np.abs(z)
    diff = np.diff(z, axis=1)
    series = {"mean": z.mean(1), "std": z.std(1), "rms": np.sqrt(np.mean(z * z, 1)), "ptp": np.ptp(z, 1),
              "abs_mean": absz.mean(1), "maxabs": absz.max(1), "diff_std": diff.std(1)}
    out = {"n": len(x)}
    w10 = max(1, windows // 10)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for name, v in series.items():
        q = np.quantile(v, [.05, .1, .25, .5, .75, .9, .95])
        out.update({f"{name}_q{i}": float(a) for i, a in enumerate(q)})
        out[f"{name}_first"] = float(v[0])
        out[f"{name}_last"] = float(v[-1])
        out[f"{name}_first10_mean"] = float(v[:w10].mean())
        out[f"{name}_last10_mean"] = float(v[-w10:].mean())
        out[f"{name}_delta"] = float(v[-1] - v[0])
        out[f"{name}_delta10"] = float(v[-w10:].mean() - v[:w10].mean())
        out[f"{name}_ratio10"] = float(v[-w10:].mean() / (abs(v[:w10].mean()) + 1e-8))
        out[f"{name}_std_time"] = float(v.std())
        out[f"{name}_max_time"] = float(v.max())
        out[f"{name}_min_time"] = float(v.min())
        out[f"{name}_slope"] = float(np.polyfit(t, v, 1)[0])
        out[f"{name}_slope_firsthalf"] = float(np.polyfit(t[:windows // 2], v[:windows // 2], 1)[0])
        out[f"{name}_slope_lasthalf"] = float(np.polyfit(t[windows // 2:], v[windows // 2:], 1)[0])
        out[f"{name}_auc"] = float(trap(v, t))
    out["ptp_over_std_mean"] = float(np.mean(series["ptp"] / (series["std"] + 1e-8)))
    out["ptp_slope_minus_std_slope"] = out["ptp_slope"] - out["std_slope"]
    return out


# ------------------------------------------------------------------- ACV ----
import re as _re

_ACV_PAT = _re.compile(r"Car (\d+) - (.*)")


def acv_case_features(path) -> pd.DataFrame:
    """Per-car, per-signal descriptors of one ACV case, centred on the case median
    (mirrors `case_rows` in nebulax src/ps3_acv_probe.py).  Columns: case, car, then
    `<signal>|mean|std|missing|delta` for every signal the case exposes."""
    from pathlib import Path
    path = Path(path)
    df = pd.read_excel(path)
    by = {}
    for col in df.columns:
        m = _ACV_PAT.fullmatch(str(col))
        if not m:
            continue
        car, s = m.groups()
        by.setdefault(car, {})[s] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for car, cols in sorted(by.items()):
        r = {"case": path.name, "car": car}
        for s, x in cols.items():
            a = x.to_numpy(float)
            valid = a[np.isfinite(a)]
            if not len(valid):
                continue
            r[f"{s}|mean"] = np.mean(valid)
            r[f"{s}|std"] = np.std(valid)
            r[f"{s}|missing"] = 1 - len(valid) / len(a)
            r[f"{s}|delta"] = valid[-1] - valid[0]
        rows.append(r)
    d = pd.DataFrame(rows)
    meta = d[["case", "car"]]
    x = d.drop(columns=["case", "car"]).apply(pd.to_numeric, errors="coerce")
    x = x.groupby(d.case).transform(lambda z: z - z.median())
    return pd.concat([meta, x], axis=1)
