"""TRM-inspired tiny recursive reasoning model.

Idea (inspired by the Tiny Recursive Model line of research): instead of one
giant network, use a SMALL reasoning block and apply it RECURSIVELY — the same
weights, reused R times — letting the latent state refine itself step by step.
Training uses deep supervision: the output head is applied after EVERY recursion
step and the loss is the average over steps, so each step is pushed to be a
better answer than the last.

The reasoning block is an MLP-Mixer style layer (token-mixing + channel-mixing
MLPs): it lets every cell attend to every other cell with plain matrix
multiplies, so the whole model runs in milliseconds on a CPU with no GPU and
no attention-kernel tricks.

This is "TRM-inspired", not a reproduction of any paper and not a SOTA claim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReasoningBlock(nn.Module):
    """One tiny mixer layer. A single instance is shared across all recursion
    steps — recursion reuses weights, it does not copy them."""

    def __init__(self, d_model: int = 48, n_tokens: int = 81, d_tok: int = 162,
                 d_ff: int = 128) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # token mixing: every cell sees every other cell
        self.tok_mix = nn.Sequential(
            nn.Linear(n_tokens, d_tok),
            nn.GELU(),
            nn.Linear(d_tok, n_tokens),
        )
        self.ln2 = nn.LayerNorm(d_model)
        # channel mixing: per-cell feature transform
        self.chan_mix = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.ln1(z + self.tok_mix(z.transpose(1, 2)).transpose(1, 2))
        return self.ln2(z + self.chan_mix(z))


class TinyRecursiveModel(nn.Module):
    """Embed -> recurse the shared block R times -> per-step classification head."""

    def __init__(
        self,
        d_model: int = 48,
        n_steps: int = 4,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.token_emb = nn.Embedding(10, d_model)  # digits 0..9 (0 = blank)
        self.pos_emb = nn.Embedding(81, d_model)  # cell positions
        self.block = ReasoningBlock(d_model)
        self.head = nn.Linear(d_model, 10)

    def reason(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the recursion; return the output logits after EACH step."""
        b = x.size(0)
        pos = torch.arange(81, device=x.device).unsqueeze(0).expand(b, -1)
        z = self.token_emb(x) + self.pos_emb(pos)
        step_logits: list[torch.Tensor] = []
        for _ in range(self.n_steps):
            z = self.block(z)  # same block, reused — this is the recursion
            step_logits.append(self.head(z))
        return step_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reason(x)[-1]

    def deep_supervision_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over all recursion steps (deep supervision)."""
        losses = [
            F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            for logits in self.reason(x)
        ]
        return torch.stack(losses).mean()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy per-cell prediction from the final recursion step."""
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
