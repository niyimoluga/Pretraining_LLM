"""Small decoder: pre-RMSNorm, RoPE, SwiGLU, tied embeddings and causal attention.

An educational PyTorch implementation, inspired by Karpathy's micro/nanoGPT and
CS336's component-based approach; not compatible with pretrained model weights.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from components import RMSNorm, rotate


@dataclass(frozen=True)
class Config:
    vocab_size: int
    context: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    activation: str = "silu"

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.layers, self.heads, self.width) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into heads with even head dimensions for RoPE")
        if self.activation not in ("silu", "gelu"):
            raise ValueError("Supported activations: silu, gelu")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, cos, sin):
        """TODO 1: causal attention. x is [B, T, C]; return [B, T, C].

        Project Q/K/V, split C into heads, apply rotate to Q and K,
        calculate scaled attention scores, mask future positions, normalize
        with softmax, combine values, merge heads and apply self.out.
        cos/sin contain positions up to the configured maximum context.
        See assignment.md for exact shapes, scaling and mask direction.
        """
        B, T, C = x.shape
        H = self.heads
        D = C // H  # width of one head; C is guaranteed divisible by H by Config

        # 1. One projection gives Q, K and V stacked on the last axis: [B, T, 3C].
        #    chunk(3) returns them in that order (q, k, v), each [B, T, C].
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # 2. Split the C channels into H heads and move the head axis forward:
        #    [B, T, C] -> [B, T, H, D] -> [B, H, T, D].
        #    Contiguity of the view is preserved because T and H*D partition the
        #    same memory (no transpose before the view).
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # 3. Rotary positions on Q and K only, using the first T rows of the
        #    tables. rotate() rotates adjacent (even, odd) pairs, so cos/sin of
        #    shape [T, D/2] broadcast across batch and head axes.
        cos_t, sin_t = cos[:T], sin[:T]
        q = rotate(q, cos_t, sin_t)
        k = rotate(k, cos_t, sin_t)

        # 4. Scaled dot-product scores: [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T].
        #    Scale by sqrt(D), the per-head dimension (not sqrt(C)).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        # 5. Causal mask: query position i may only attend to key positions j <= i.
        #    tril(ones) is exactly the allowed set; everything above the diagonal
        #    becomes -inf so softmax gives it zero weight.
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))

        # 6. Softmax over the last (key) axis. Do it in float32 for numerical
        #    stability, then cast back to the values' dtype (supports bf16).
        weights = scores.float().softmax(dim=-1).to(v.dtype)

        # 7. Weighted sum of values -> [B, H, T, D].
        y = weights @ v

        # 8. Merge heads back into the model width: [B, H, T, D] -> [B, T, H, D]
        #    -> [B, T, C], then mix channels with the output projection.
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = config.activation
        self.attn_norm, self.ffn_norm = RMSNorm(config.width), RMSNorm(config.width)
        self.attn = Attention(config)
        hidden = 64 * math.ceil((8 * config.width / 3) / 64)
        self.gate = nn.Linear(config.width, hidden, bias=False)
        self.up = nn.Linear(config.width, hidden, bias=False)
        self.down = nn.Linear(hidden, config.width, bias=False)

    def forward(self, x, cos, sin):
        """TODO 2: connect the supplied layers into a transformer block.

        Normalize before each sublayer. Keep both residual connections.
        Use self.activation to select silu (baseline) or gelu (easy variant).
        See assignment.md for the gated feed-forward equation.
        """
        # Pre-norm attention sublayer with a residual connection:
        #     x = x + attention(attn_norm(x))
        x = x + self.attn(self.attn_norm(x), cos, sin)

        # Pre-norm gated feed-forward sublayer (SwiGLU when activation is silu):
        #     h = ffn_norm(x)
        #     x = x + down(activation(gate(h)) * up(h))
        # Both gate and up read the same normalized input; the element-wise
        # product is the gate that decides how much of each up-projection unit
        # passes. The activation is the only difference between the baseline
        # (SiLU) and the variant (GELU), selected at construction from Config.
        h = self.ffn_norm(x)
        activation = F.silu if self.activation == "silu" else F.gelu
        x = x + self.down(activation(self.gate(h)) * self.up(h))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        head_dim = config.width // config.heads
        angles = torch.outer(torch.arange(config.context).float(),
                             10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        # Scale residual projections to keep initial residual updates small at depth.
        for block in self.blocks:
            for projection in (block.attn.out, block.down):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.layers))

    def forward(self, ids, targets=None):
        """TODO 3: token IDs -> logits and optional next-token loss.

        ids: integer [B, T], 1 <= T <= context.
        targets: optional integer [B, T]; -100 means ignore this target.
        Return (logits [B, T, vocab_size], scalar loss or None).
        Use the blocks, final RMSNorm and tied embedding/output weights.
        The caller already shifts the targets; do not shift them again.
        """
        # Reject malformed inputs early. The model only supports [B, T] integer
        # sequences whose length fits the position tables (1 <= T <= context).
        if ids.ndim != 2:
            raise ValueError(f"ids must be two-dimensional [B, T], got shape {tuple(ids.shape)}")
        T = ids.shape[1]
        if not 1 <= T <= self.config.context:
            raise ValueError(f"sequence length must satisfy 1 <= T <= {self.config.context}, got {T}")

        # Token embeddings -> transformer blocks -> final normalization.
        # cos/sin hold positions up to the maximum context; the blocks slice [:, :T].
        x = self.embedding(ids)                       # [B, T, C]
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.norm(x)                              # [B, T, C]

        # Tied output projection: reuse the embedding matrix as the unembedding.
        # embedding.weight is [vocab, C], so x @ weight.T gives [B, T, vocab].
        logits = x @ self.embedding.weight.T          # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Targets are already shifted by the caller. Compute the mean
            # cross-entropy from float32 logits, ignoring targets equal to -100
            # (padding and other unscored positions). Use reshape (not view):
            # make_targets returns a non-contiguous slice windows[:, 1:], and
            # view(-1) rejects non-contiguous input while reshape handles it.
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, count, temperature=0.8, top_k=40, stop=None):
        if not ids or count < 0 or temperature < 0 or top_k < 0:
            raise ValueError("Need a nonempty prompt and nonnegative sampling arguments")
        self.eval()
        tokens = torch.tensor([ids], device=self.embedding.weight.device)
        for _ in range(count):
            logits = self(tokens[:, -self.config.context:])[0][:, -1].float()
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                next_id = torch.multinomial(logits.softmax(-1), 1)
            tokens = torch.cat((tokens, next_id), dim=1)
            if next_id.item() == stop:
                break
        return tokens[0].tolist()
