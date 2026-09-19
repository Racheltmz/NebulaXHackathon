"""Baseline Door classifier: frozen TimesFM3 embeddings + trainable
gated-attention MIL pooling head, trained via backprop on GPU.

DEEP LEARNING ONLY track -- no sklearn/classical models (see problem.yaml).
This baseline: (1) extracts frozen TimesFM3 variate-attention embeddings for
each raw [16, T] door-cycle segment, (2) pools channels via a small trainable
gated-attention MIL layer (Ilse et al. 2018), (3) classifies with a 2-layer
MLP head, all trained end-to-end with Adam directly inside fit(). Runs on a
GPU compute node on the NUS SoC cluster; TIMESFM_SRC below is that cluster's
fixed path and only resolves there.

Contract (frozen): Model.__init__(self), Model.fit(self, X, y),
Model.predict(self, X). X is a list of raw [16, T] float arrays; y holds raw
string labels ("Normal"/"Abnormal resistance") during fit; predict() gets
only X.
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
        self.temporal = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        # Normalize each token so attention is driven by structure rather
        # than arbitrary backbone activation scale.
        tok = torch.nn.functional.layer_norm(tok, (tok.shape[-1],))

        # Learn informative patches while retaining global context.
        temporal_logits = self.temporal(tok).squeeze(-1)
        temporal_w = torch.softmax(temporal_logits, dim=2)
        temporal_w = 0.7 * temporal_w + 0.3 / tok.shape[2]
        temporal_w = temporal_w.unsqueeze(-1)
        mean_p = (tok * temporal_w).sum(2)
        max_p = tok.amax(2)
        std_p = tok.std(2, unbiased=False)
        pooled = torch.cat([mean_p, max_p, std_p], dim=-1)

        # Then learn which sensor channels are informative.
        # Use both average and peak temporal responses when selecting channels.
        gate = torch.tanh(self.V(mean_p)) * torch.sigmoid(self.U(max_p))
        channel_w = torch.softmax(self.w(gate), dim=1)
        return (pooled * channel_w).sum(1)


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tfm3 = None
        self._pool = None
        self._head = None
        self._mean = None
        self._std = None
        self._labels = None

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
        y_list = y.tolist() if hasattr(y, "tolist") else list(y)
        self._labels = sorted(set(y_list))
        label_to_idx = {v: i for i, v in enumerate(self._labels)}
        y_idx = torch.tensor([label_to_idx[v] for v in y_list])
        n_out = len(self._labels)

        feats = self._extract(X).to(self.device)
        dim = feats.shape[-1]

        torch.manual_seed(17)
        self._pool = _MILPool(dim).to(self.device)
        self._head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 256), nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 96), nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, n_out),
        ).to(self.device)
        params = list(self._pool.parameters()) + list(self._head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

        # LayerNorm in the classifier handles feature scale. Keeping a
        # normalization statistic computed from the randomly initialized pool
        # would become stale as the attention pool learns.
        self._mean = None
        self._std = None

        counts = torch.bincount(y_idx, minlength=n_out).float().clamp_min(1)
        # Softer reweighting avoids overcorrecting minority labels.
        class_weight = (counts.sum() / (n_out * counts)).sqrt().to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.02)

        y_idx = y_idx.to(self.device)
        n = feats.shape[0]
        bs = min(16, max(2, n))
        for _ in range(150):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = feats[idx] + 0.05 * torch.randn_like(feats[idx])
                mask = (torch.rand(xb.shape[:3], device=self.device) > 0.15).float().unsqueeze(-1)
                xb = xb * mask / 0.85
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
        self._pool.eval()
        # Monte Carlo dropout provides a cheap neural snapshot ensemble at
        # inference time and reduces sensitivity to one randomly thinned head.
        self._head.train()
        with torch.no_grad():
            pooled = self._pool(feats)
            probs = 0.0
            for _ in range(8):
                probs = probs + torch.softmax(self._head(pooled), dim=-1)
            probs = probs / 8.0
            idx = probs.argmax(-1).cpu().numpy()
        self._head.eval()
        return np.array([self._labels[i] for i in idx], dtype=object)
# EVOLVE-BLOCK-END
