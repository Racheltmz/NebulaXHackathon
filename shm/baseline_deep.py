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
        pooled = torch.cat([mean_p, max_p], -1)
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
            a = a[:, -context:]
            if a.shape[1] < context:
                a = np.pad(a, ((0, 0), (context - a.shape[1], 0)))
            arrs.append(a)
        feats = []
        with torch.inference_mode():
            for start in range(0, len(arrs), 16):
                chunk = arrs[start:start + 16]
                batch = np.stack(chunk)
                values = torch.from_numpy(batch).to(self.device).reshape(len(chunk), batch.shape[1], n, patch)
                masks = torch.zeros_like(values, dtype=torch.bool)
                target = torch.ones((len(chunk), batch.shape[1], n), device=self.device, dtype=torch.bool)
                out = self._tfm3(
                    {"values": values, "masks": masks, "patch_is_target": target},
                    return_aux_outputs=True,
                )
                feats.append(out["__call__:transformer_output"].float().cpu())
        return torch.cat(feats, dim=0)  # [N, C, n_patches, dim]

    def fit(self, X: Sequence[np.ndarray], y: np.ndarray) -> None:
        y_t = torch.tensor(np.asarray(y, dtype=np.float32))
        feats = self._extract(X).to(self.device)
        dim = feats.shape[-1]

        torch.manual_seed(17)
        self._pool = _MILPool(dim).to(self.device)
        self._head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(2 * dim, 128), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(128, 1),
        ).to(self.device)
        params = list(self._pool.parameters()) + list(self._head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

        with torch.no_grad():
            pooled0 = self._pool(feats)
            self._mean = pooled0.mean(0, keepdim=True)
            self._std = pooled0.std(0, keepdim=True).clamp_min(1e-4)

        y_log = torch.log(y_t.clamp_min(1e-5))
        self._y_mean = y_log.mean()
        self._y_std = y_log.std().clamp_min(1e-4)
        y_scaled = ((y_log - self._y_mean) / self._y_std).to(self.device).unsqueeze(-1)

        n = feats.shape[0]
        bs = min(16, max(2, n))
        for _ in range(200):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = feats[idx] + 0.05 * torch.randn_like(feats[idx])
                pooled = (self._pool(xb) - self._mean) / self._std
                pred = self._head(pooled)
                loss = nn.functional.smooth_l1_loss(pred, y_scaled[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2.0)
                opt.step()
            sched.step()

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        feats = self._extract(X).to(self.device)
        with torch.no_grad():
            pooled = (self._pool(feats) - self._mean) / self._std
            pred = self._head(pooled).squeeze(-1)
            pred = torch.exp(pred * self._y_std + self._y_mean)
        return pred.cpu().numpy().astype(float)
# EVOLVE-BLOCK-END
