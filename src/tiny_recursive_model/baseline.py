"""The honest comparison: a BIGGER, non-recursive feed-forward model.

Same task, same data, same training budget — but ~150x more parameters and no
recursion. If the tiny recursive model wins (or ties), the win comes from the
recursive reasoning, not from parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BigBaseline(nn.Module):
    """Embed cells -> flatten -> deep MLP -> per-cell logits. No recursion."""

    def __init__(self, d_emb: int = 24, hidden: int = 1024, n_layers: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(10, d_emb)
        layers: list[nn.Module] = []
        dim = 81 * d_emb
        for _ in range(n_layers):
            layers += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 81 * 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.emb(x).reshape(x.size(0), -1)
        return self.head(self.net(z)).reshape(-1, 81, 10)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)
