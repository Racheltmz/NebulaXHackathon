"""Baseline ACV fault-ranking model: frozen TimesFM3 embeddings + trainable
gated-attention MIL pooling head, trained via backprop on GPU.

DEEP LEARNING ONLY track -- no sklearn/classical models (see problem.yaml).
Trains a binary (faulty/not) classifier internally, but predict() returns
the continuous P(faulty) score, not a hard label -- the harness ranks cars
within each held-out case by this score and scores by the official
rank-decay formula. Cars are not explicitly grouped by case in X/y; case
membership is a harness-only concept used for splitting/scoring.

Contract (frozen): Model.__init__(self), Model.fit(self, X, y),
Model.predict(self, X). X is a list of raw [8, T] float arrays, one per car
(4 raw per-timestep stats + 4 case-relative residual stats -- see
ps3_adapter.py's acv()). y is 1 for the faulty car, 0 otherwise, during fit.
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
    """Channel attention with complementary temporal statistics."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        mean_p = tok.mean(2)
        max_p = tok.amax(2)
        gate = torch.tanh(self.V(mean_p)) * torch.sigmoid(self.U(mean_p))
        weights = torch.softmax(self.w(gate), dim=1)
        std_p = tok.std(2, unbiased=False)
        last_p = tok[:, :, -1]
        stats = torch.stack((mean_p, max_p, std_p, last_p), dim=2)
        return (stats * weights.unsqueeze(-1)).sum(1).flatten(1)


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tfm3 = None
        self._pool = None
        self._head = None
        self._mean = None
        self._std = None

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
        y_idx = torch.tensor(np.asarray(y, dtype=np.int64))
        feats = self._extract(X).to(self.device)
        dim = feats.shape[-1]

        torch.manual_seed(17)
        self._pool = _MILPool(dim).to(self.device)
        self._head = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Dropout(0.25), nn.Linear(4 * dim, 96), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(96, 2),
        ).to(self.device)
        params = list(self._pool.parameters()) + list(self._head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

        # LayerNorm in the head provides stable feature scaling; avoid using
        # stale pool statistics while the attention pool is still adapting.
        self._mean = torch.zeros((1, 4 * dim), device=self.device)
        self._std = torch.ones((1, 4 * dim), device=self.device)

        counts = torch.bincount(y_idx, minlength=2).float().clamp_min(1)
        class_weight = (counts.sum() / (2 * counts)).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        y_idx = y_idx.to(self.device)
        n = feats.shape[0]
        bs = min(16, max(2, n))
        for _ in range(150):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = feats[idx] + 0.03 * torch.randn_like(feats[idx])
                pooled = self._pool(xb)
                logits = self._head(pooled)
                loss = loss_fn(logits, y_idx[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2.0)
                opt.step()
            sched.step()

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        feats = self._extract(X).to(self.device)
        with torch.no_grad():
            # Mild feature-space TTA matches the augmentation used in training
            # and reduces variance in the ranking scores.
            logits = 0.0
            for _ in range(4):
                pooled = self._pool(feats + 0.012 * torch.randn_like(feats))
                logits = logits + self._head(pooled)
            score = torch.softmax(logits / 4.0, -1)[:, 1]
        return score.cpu().numpy().astype(float)
# EVOLVE-BLOCK-END
