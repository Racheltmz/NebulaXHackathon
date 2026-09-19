"""Baseline SHM fatigue-damage regressor: frozen TimesFM3 embeddings +
trainable gated-attention MIL pooling head, trained via backprop on GPU.

DEEP LEARNING ONLY track -- no sklearn/classical models (see problem.yaml).
Note: TimesFM3's context window (512 timesteps) is far shorter than SHM's
raw signal length (~581k samples), so this baseline only sees the last 512
samples through the frozen backbone -- there is real room to improve this,
e.g. by having predict()/fit() compute auxiliary full-signal statistics
(peak/RMS/rainflow cycle counting, still just numpy -- that's feature
engineering feeding a neural net, not a classical model, and is allowed)
and feeding them into the trainable head alongside the pooled embedding.

Contract (frozen): Model.__init__(self), Model.fit(self, X, y),
Model.predict(self, X). X is a list of raw [1, T] float arrays; y holds raw
float cumulative-damage targets during fit; predict() gets only X and must
return one float per row.
"""
from __future__ import annotations

import sys
from typing import Sequence

import numpy as np
import torch
from torch import nn

TIMESFM_SRC = "/home/s/saad304/timesfm/src"
CHECKPOINT = "google/timesfm-3.0-pytorch"


# EVOLVE-BLOCK-START
class _MILPool(nn.Module):
    """Gated-attention MIL pooling over channels-as-instances. Underscore-
    prefixed and inside the evolve block on purpose: it's a starting point,
    not part of the frozen contract -- feel free to replace, remove, or
    redesign it entirely, as long as `Model.fit`/`Model.predict` keep their
    signatures."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        mean_p = tok.mean(2)
        max_p = tok.amax(2)
        std_p = tok.std(2, unbiased=False)
        pooled = torch.cat([mean_p, max_p, std_p], -1)
        gate = torch.tanh(self.V(mean_p)) * torch.sigmoid(self.U(mean_p))
        w = torch.softmax(self.w(gate), dim=1)
        return (pooled * w).sum(1)


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tfm3 = None
        self._pool = None
        self._head = None
        self._mean = None
        self._std = None
        self._y_mean = None
        self._y_std = None
        self._rmean = None
        self._rstd = None

    def _load_backbone(self) -> None:
        if self._tfm3 is not None:
            return
        if TIMESFM_SRC not in sys.path:
            sys.path.insert(0, TIMESFM_SRC)
        from timesfm3.torch.model import TimesFM3Torch
        model = TimesFM3Torch.from_pretrained(CHECKPOINT).to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._tfm3 = model

    def _extract(self, X: Sequence[np.ndarray]) -> torch.Tensor:
        self._load_backbone()
        context = 512
        patch = self._tfm3.input_patch_len
        n = context // patch
        arrs = []
        for x in X:
            a = np.nan_to_num(np.asarray(x, dtype=np.float32))
            if a.ndim == 1:
                a = a[None, :]
            a = (a - a.mean(axis=1, keepdims=True)) / np.maximum(a.std(axis=1, keepdims=True), 1e-5)
            T = a.shape[1]
            starts = [max(0, T - context), max(0, (T - context) // 2), 0]
            views = []
            for s in starts:
                v = a[:, s:s + context]
                if v.shape[1] < context:
                    v = np.pad(v, ((0, 0), (context - v.shape[1], 0)))
                views.append(v)
            arrs.append(np.concatenate(views, axis=1))
        feats = []
        with torch.inference_mode():
            for start in range(0, len(arrs), 16):
                chunk = arrs[start:start + 16]
                batch = np.stack(chunk)
                values = torch.from_numpy(batch).to(self.device).reshape(
                    len(chunk), batch.shape[1], 3 * n, patch)
                masks = torch.zeros_like(values, dtype=torch.bool)
                target = torch.ones((len(chunk), batch.shape[1], 3 * n),
                                    device=self.device, dtype=torch.bool)
                out = self._tfm3(
                    {"values": values, "masks": masks, "patch_is_target": target},
                    return_aux_outputs=True,
                )
                feats.append(out["__call__:transformer_output"].float().cpu())
        return torch.cat(feats, dim=0)

    def _raw_features(self, X: Sequence[np.ndarray]) -> torch.Tensor:
        out = []
        for x in X:
            a = np.nan_to_num(np.asarray(x, dtype=np.float32))
            if a.ndim == 1:
                a = a[None, :]
            z = a.reshape(-1)
            d = np.diff(z) if z.size > 1 else z
            az = np.abs(z)
            ad = np.abs(d)
            q = np.percentile(z, [1, 5, 25, 50, 75, 95, 99])
            aq = np.percentile(az, [90, 99])
            n = z.size
            cuts = np.linspace(0, n, 5, dtype=np.int64)
            sr = [
                np.sqrt(np.mean(z[cuts[i]:cuts[i + 1]] ** 2))
                if cuts[i + 1] > cuts[i] else 0.0
                for i in range(4)
            ]
            rms = np.sqrt(np.mean(z * z))
            drms = np.sqrt(np.mean(d * d))
            m = min(4096, n)
            sample = z[np.linspace(0, n - 1, m).astype(np.int64)]
            power = np.abs(np.fft.rfft(sample - sample.mean()))[1:] ** 2
            total = power.sum() + 1e-8
            bands = [float(v.sum() / total) for v in np.array_split(power, 4)]
            centroid = float(
                (power * np.arange(1, len(power) + 1)).sum()
                / (total * max(1, len(power)))
            )
            prob = power / total
            entropy = float(
                -(prob * np.log(prob + 1e-12)).sum()
                / np.log(max(2, len(prob)))
            )
            out.append([
                np.mean(z), np.std(z), rms, np.mean(az), np.max(az),
                q[0], q[1], q[2], q[3], q[4], q[5], q[6],
                np.ptp(z), drms, np.sqrt(np.mean(d * d * d * d)),
                np.mean(ad > 2.0 * (np.std(d) + 1e-6)),
                np.mean(z[:-1] * z[1:] < 0) if n > 1 else 0.0,
                *sr, np.log1p(rms), np.log1p(aq[0]), np.log1p(aq[1]),
                np.max(az) / (rms + 1e-6), np.mean(ad),
                *bands, centroid, entropy,
            ])
        return torch.tensor(np.asarray(out, dtype=np.float32), device=self.device)

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        y_t = torch.tensor(np.asarray(y, dtype=np.float32))
        feats = self._extract(X).to(self.device)
        raw = self._raw_features(X)
        dim = feats.shape[-1]

        torch.manual_seed(17)
        self._pool = _MILPool(dim).to(self.device)
        self._head = nn.Sequential(
            nn.LayerNorm(3 * dim + raw.shape[1]),
            nn.Linear(3 * dim + raw.shape[1], 256), nn.GELU(),
            nn.Dropout(0.12), nn.Linear(256, 96), nn.GELU(),
            nn.Linear(96, 1),
        ).to(self.device)
        params = list(self._pool.parameters()) + list(self._head.parameters())
        opt = torch.optim.AdamW(params, lr=7e-4, weight_decay=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=240)

        with torch.no_grad():
            self._mean = torch.zeros((1, 3 * dim), device=self.device)
            self._std = torch.ones((1, 3 * dim), device=self.device)
            self._rmean = raw.mean(0, keepdim=True)
            self._rstd = raw.std(0, keepdim=True).clamp_min(1e-4)

        y_log = torch.log(y_t.clamp_min(1e-5))
        self._y_mean = y_log.mean()
        self._y_std = y_log.std().clamp_min(1e-4)
        y_scaled = ((y_log - self._y_mean) / self._y_std).to(self.device).unsqueeze(-1)

        n = feats.shape[0]
        bs = min(16, max(2, n))
        for _ in range(240):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = feats[idx] + 0.025 * torch.randn_like(feats[idx])
                pooled = (self._pool(xb) - self._mean) / self._std
                rb = (raw[idx] - self._rmean) / self._rstd
                pred = self._head(torch.cat([pooled, rb], dim=-1))
                loss = nn.functional.smooth_l1_loss(pred, y_scaled[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2.0)
                opt.step()
            sched.step()

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        feats = self._extract(X).to(self.device)
        raw = self._raw_features(X)
        with torch.no_grad():
            pooled = (self._pool(feats) - self._mean) / self._std
            raw = (raw - self._rmean) / self._rstd
            pred = self._head(torch.cat([pooled, raw], dim=-1)).squeeze(-1)
            pred = torch.exp(pred * self._y_std + self._y_mean)
        return pred.cpu().numpy().astype(float)
# EVOLVE-BLOCK-END
