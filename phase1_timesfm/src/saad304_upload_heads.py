import torch
from torch import nn


class AttentionHead(nn.Module):
    """FORMED-style shared decoder with dataset-local channel/class tokens."""

    def __init__(self, dim: int, max_classes: int = 32, queries_per_class: int = 4,
                 heads: int = 8, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.qpc = queries_per_class
        self.max_classes = max_classes
        self.channel_proj = nn.Linear(dim, dim, bias=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout,
                                       batch_first=True, norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, features, channel_embeddings, label_queries):
        # features: [B,C,L,D], channel_embeddings: [C,D], label_queries: [K*qpc,D]
        b, c, l, d = features.shape
        memory = features + channel_embeddings[None, :, None, :]
        memory = self.channel_proj(memory).reshape(b, c * l, d)
        q = label_queries[None].expand(b, -1, -1)
        decoded = self.decoder(q, memory)
        raw = self.out(decoded).squeeze(-1).view(b, -1, self.qpc)
        return raw.mean(-1)


class PoolHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, queries_per_class: int = 4):
        super().__init__()
        self.qpc = queries_per_class
        self.net = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, dim))

    def forward(self, features, channel_embeddings, label_queries):
        x = features + channel_embeddings[None, :, None, :]
        x = torch.cat([x.mean((1, 2)), x.std((1, 2), unbiased=False)], -1)
        z = self.net(x)
        # A shared metric against class queries keeps the head class-agnostic.
        logits = (z @ label_queries.T) / (z.shape[-1] ** 0.5)
        return logits.view(logits.shape[0], -1, self.qpc).mean(-1)


class LinearHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, queries_per_class: int = 4):
        super().__init__()
        self.qpc = queries_per_class
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.out = nn.Linear(hidden, dim)

    def forward(self, features, channel_embeddings, label_queries):
        x = self.proj((features + channel_embeddings[None, :, None, :]).mean((1, 2)))
        logits = self.out(x) @ label_queries.T / (x.shape[-1] ** 0.5)
        return logits.view(logits.shape[0], -1, self.qpc).mean(-1)

class LowRankLinearHead(nn.Module):
    """Class-agnostic linear head with a compact shared task embedding."""
    def __init__(self, dim: int, embed: int = 128, hidden: int = 256, queries_per_class: int = 4):
        super().__init__(); self.qpc=queries_per_class; self.query_dim=embed
        self.proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,embed),nn.LayerNorm(embed))
    def forward(self, features, channel_embeddings, label_queries):
        x=self.proj((features+channel_embeddings[None,:,None,:]).mean((1,2)))
        logits=(x @ label_queries.T)/(x.shape[-1]**0.5)
        return logits.view(logits.shape[0],-1,self.qpc).mean(-1)


class RegressionHead(nn.Module):
    """Scalar score head for SHM degradation or pairwise reranking."""
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, features, channel_embeddings, label_queries=None):
        x = features + channel_embeddings[None, :, None, :]
        x = torch.cat([x.mean((1, 2)), x.std((1, 2), unbiased=False)], -1)
        return self.net(x).squeeze(-1)


def make_head(variant, dim, qpc=4):
    if variant == "regression":
        return RegressionHead(dim)
    if variant == "pool":
        return PoolHead(dim, queries_per_class=qpc)
    if variant == "linear":
        return LinearHead(dim, queries_per_class=qpc)
    if variant == "linear128":
        return LowRankLinearHead(dim, embed=128, queries_per_class=qpc)
    return AttentionHead(dim, queries_per_class=qpc, layers=2 if variant == "sda_big" else 1)
