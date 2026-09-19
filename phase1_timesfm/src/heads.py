import sys
from pathlib import Path

import torch
from torch import nn

_TIMESFM_SRC = Path("/home/s/saad304/timesfm/src")
if _TIMESFM_SRC.exists() and str(_TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(_TIMESFM_SRC))
try:
    from timesfm3.torch import configs as tfm3_configs
    from timesfm3.torch.transformer import MixingTransformer
except Exception:
    tfm3_configs = None
    MixingTransformer = None


def _class_logits(z, label_queries, qpc):
    logits = (z @ label_queries.T) / (z.shape[-1] ** 0.5)
    return logits.view(logits.shape[0], -1, qpc).mean(-1)


class LinearHead(nn.Module):
    """Shared pooled metric head used as a low-capacity baseline."""
    def __init__(self, dim: int, hidden: int = 256, queries_per_class: int = 4):
        super().__init__()
        self.qpc = queries_per_class
        self.query_dim = dim
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.out = nn.Linear(hidden, dim)

    def forward(self, features, channel_embeddings, label_queries):
        x = self.proj((features + channel_embeddings[None, :, None, :]).mean((1, 2)))
        return _class_logits(self.out(x), label_queries, self.qpc)


class FlamingoExactHead(nn.Module):
    """Shared-dataset form of the public FlaMinGo classifier architecture.

    The released FlaMinGo wrapper uses mean TimesFM transformer features and
    Linear(1280, hidden_dim=160) -> ReLU -> Dropout(.65) -> Linear(hidden_dim,
    num_classes). The final dataset-specific Linear is represented here by
    label queries, so one shared projector can be trained across datasets with
    different class counts.
    """
    def __init__(self, dim: int, hidden: int = 160, queries_per_class: int = 1):
        super().__init__()
        self.qpc = queries_per_class
        self.query_dim = hidden + 1
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p=0.65)
        )

    def forward(self, features, channel_embeddings, label_queries):
        x = features + channel_embeddings[None, :, None, :]
        z = self.classifier(x.mean((1, 2)))
        z = torch.cat([z, torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)], dim=-1)
        raw = z @ label_queries.T
        return raw.view(raw.shape[0], -1, self.qpc).mean(-1)


class PaperSDAHead(nn.Module):
    """FORMED/SDA-faithful one-layer Transformer decoder classifier.

    The paper specifies channel embeddings, label queries, one shared
    Transformer decoder layer, FFN processing, and averaging k raw logits per
    class. The decoder uses PyTorch's standard self-attention, cross-attention,
    residual/normalization, and 4D feed-forward sublayers.
    """
    def __init__(self, dim: int, queries_per_class: int = 16,
                 heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.qpc = queries_per_class
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=False
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.out = nn.Linear(dim, 1)

    def forward(self, features, channel_embeddings, label_queries):
        b, c, l, d = features.shape
        memory = (features + channel_embeddings[None, :, None, :]).reshape(b, c * l, d)
        queries = label_queries[None].expand(b, -1, -1)
        decoded = self.decoder(queries, memory)
        raw = self.out(decoded).squeeze(-1)
        return raw.view(b, -1, self.qpc).mean(-1)


class TimesFM3HybridSDAHead(nn.Module):
    """TimesFM3 MixingTransformer block followed by paper SDA decoding.

    The refinement block is the actual TimesFM3 implementation: causal
    sequence attention, full variate attention at each patch, and its RMSNorm
    residual FFN. It is followed by the same one-layer decoder/logit path as
    PaperSDAHead.
    """
    def __init__(self, dim: int, queries_per_class: int = 16,
                 heads: int = 16, dropout: float = 0.0):
        super().__init__()
        if MixingTransformer is None or tfm3_configs is None:
            raise ImportError("TimesFM3 source is unavailable on the remote cluster")
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.qpc = queries_per_class
        tfm_cfg = tfm3_configs.TransformerConfig(
            model_dims=dim, hidden_dims=dim, num_heads=heads,
            attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
            use_bias=False, use_rope_seq=True, use_rope_var=False,
            ff_activation="relu", deterministic=True, causal_attention=True,
            training=False, use_memory_efficient_attention=True, use_sdpa=True
        )
        self.tfm3_mix = MixingTransformer(tfm_cfg, use_variate_attention=True)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=8, dim_feedforward=4 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=False
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.out = nn.Linear(dim, 1)

    def forward(self, features, channel_embeddings, label_queries):
        b, c, l, d = features.shape
        x = features + channel_embeddings[None, :, None, :]
        patch_mask = torch.zeros((b, c, l), dtype=torch.bool, device=x.device)
        x, _, _ = self.tfm3_mix(x, patch_mask)
        memory = x.reshape(b, c * l, d)
        queries = label_queries[None].expand(b, -1, -1)
        decoded = self.decoder(queries, memory)
        raw = self.out(decoded).squeeze(-1)
        return raw.view(b, -1, self.qpc).mean(-1)


def make_head(variant, dim, qpc=4):
    if variant == "linear":
        return LinearHead(dim, queries_per_class=qpc)
    if variant in ("flamingo", "flamingo_exact"):
        return FlamingoExactHead(dim, queries_per_class=qpc)
    if variant in ("sda", "paper_sda"):
        return PaperSDAHead(dim, queries_per_class=qpc)
    if variant in ("hybrid", "timesfm3_sda"):
        return TimesFM3HybridSDAHead(dim, queries_per_class=qpc)
    raise ValueError(f"Unknown head variant: {variant}")
