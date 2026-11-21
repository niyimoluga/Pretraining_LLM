# TODO 2 — `Block.forward`: structure, verification and defence notes

Implemented 2026-11-20 in `assignment1/code/model.py`.

## What the block does (pre-norm, two residual branches)

```python
x = x + self.attn(self.attn_norm(x), cos, sin)      # attention sublayer
h = self.ffn_norm(x)
activation = F.silu if self.activation == "silu" else F.gelu
x = x + self.down(activation(self.gate(h)) * self.up(h))   # gated FFN sublayer
return x
```

This is exactly the equation in `assignment.md`:

```text
x = x + attention(attn_norm(x))
h = ffn_norm(x)
x = x + down(activation(gate(h)) * up(h))
```

## Why each choice (defence points)

- **Pre-norm.** Both sublayers normalize their *input* (`attn_norm(x)`, `ffn_norm(x)`),
  not their output. This keeps activations well-scaled at depth and is why the residual
  path is a clean identity.
- **Two residual connections.** One around attention, one around the FFN. Removing either
  makes deep stacks untrainable; the init in `GPT` scales `attn.out` and `down` by
  `1/sqrt(2·layers)` specifically to keep these residual updates small at the start.
- **Gated feed-forward (SwiGLU).** `gate` and `up` both read the same `h`; the element-wise
  product `activation(gate(h)) · up(h)` lets the gate modulate how much of each
  up-projection channel passes. Hidden width is `64·ceil((8C/3)/64)` (384 for C=128,
  1408 for C=512), which keeps the FFN parameter count roughly constant regardless of
  the gate style.
- **Activation chosen at construction** from `Config.activation` (`silu` or `gelu`), so the
  baseline and the ablation differ in exactly one code path and nothing else.

## Verification (scratch check, results reproduced)

| Check | Result |
|---|---|
| output shape `[B,T,C] -> [B,T,C]` | `(2, 8, 16)` ✅ |
| zeroing `qkv`, `out` and `down` ⇒ output equals input (residual identity) | **True** |
| output equals the hand-written equation with `F.silu` | **True** |
| output equals the hand-written equation with `F.gelu` | **True** |
| SiLU vs GELU actually differ (same weights, only activation changed) | **True**, max diff 5.9e-02 |
| GELU and SiLU have identical parameter count | **True** |

## Parameter counts (for the report)

| Config | Structure | Parameters |
|---|---|---|
| laptop | 2 layers, width 128, ctx 128, 4 heads, vocab 4096 | **950,912** (~0.95M) |
| final | 8 layers, width 512, ctx 512, 8 heads, vocab 4096 | **27,795,968** (~27.8M) |

Embedding is tied (input and output share one tensor): 524,288 params for vocab 4096 × 128
(laptop), 2,097,152 for 4096 × 512 (final). **GELU does not change the parameter count or
the FLOPs** — it is a pure activation swap, which is what makes the ablation clean.

Per-block hidden sizes: width 128 → 384; width 512 → 1408. Per-block params: 213,248
(width 128) and 3,212,288 (width 512).
