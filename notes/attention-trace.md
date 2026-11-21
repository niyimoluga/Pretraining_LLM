# TODO 1 — `Attention.forward`: shapes, verification and defence notes

Implemented 2026-11-20 in `assignment1/code/model.py`.

## The data flow (with a concrete example)

Config used for the trace: `B=2, T=8, C=16, H=2` → head width `D = C/H = 8`.

| Step | Operation | Shape |
|---|---|---|
| 0 | input `x` | `[2, 8, 16]` |
| 1 | `qkv = self.qkv(x)` | `[2, 8, 48]` |
| 1 | `q, k, v = qkv.chunk(3, dim=-1)` | three × `[2, 8, 16]` |
| 2 | `q.view(B,T,H,D).transpose(1,2)` | `[2, 2, 8, 8]` (same for k, v) |
| 3 | `rotate(q, cos[:T], sin[:T])` | `[2, 2, 8, 8]` — cos/sin are `[8, 4]`, broadcast over batch+heads |
| 4 | `q @ k.transpose(-2,-1) / sqrt(D)` | `[2, 2, 8, 8]` |
| 5 | `masked_fill(~tril, -inf)` | `[2, 2, 8, 8]` |
| 6 | `softmax(-1).to(v.dtype)` | `[2, 2, 8, 8]` |
| 7 | `weights @ v` | `[2, 2, 8, 8]` |
| 8 | `transpose(1,2).reshape(B,T,C)` then `self.out` | `[2, 8, 16]` |

Parameters in this attention: `2 × (16×48 + 16×16) = 1,024` (qkv has no bias, out has no bias).

## Why each design choice (defence points)

- **One `qkv` projection, then `chunk(3)`.** The supplied layer produces `3C` outputs;
  `chunk` splits them as q, k, v in that order. They must not be interleaved.
- **`view(B,T,H,D).transpose(1,2)`.** RoPE rotates adjacent `(even, odd)` pairs across
  the last axis, so the head width must be the last dimension when `rotate` runs.
  The `view` is valid because `T·H·D` is the same memory block as `T·C`.
- **RoPE on Q and K only.** V is not rotated — rotary positions change the *geometry of
  the query/key matching*, not the content that gets averaged.
- **Scale by `sqrt(D) = sqrt(C/H)`.** Using `sqrt(C)` would over-flatten the softmax.
- **Mask direction.** `tril` = allow `j <= i`. Everything strictly above the diagonal is
  set to `-inf` before softmax, so it receives exactly zero weight.
- **Softmax in float32, then cast.** Matches the assignment (and the norm component,
  which also accumulates in float32) for stability under bfloat16.
- **Merge order.** `transpose(1,2)` puts heads back next to the sequence axis, then
  `reshape` interleaves head channels contiguously as the `out` layer expects.

## Independent verification (scratch check, not committed as a test)

Compared the vectorized attention against a naive per-`(batch, head, query)` loop that
builds each row with an explicit `j <= i` slice:

| Check | Result |
|---|---|
| max absolute difference vs naive reference | **5.96e-08** (float32 rounding) |
| causality: past outputs unchanged after perturbing future tokens | **True** |
| future positions do change (mask not over-restrictive) | **True** |
| RoPE actually applied (output differs from identity rotation) | **True** |
| softmax rows sum to 1 | **True** |
| strictly upper triangle is exactly 0 | **True** |
| `rotate` preserves vector norm / is identity at position 0 | **True** |

### One subtlety worth remembering for the defence

With *identical* tokens at positions 0 and 1, the attention outputs at those two
positions are equal even with RoPE. That is correct, not a bug: rotation preserves dot
products, so `rot(q,1)·k = rot(q,1)·rot(k,1) = q·k`, the two scores tie at 0.5/0.5, and
since V is unrotated the weighted sum collapses to `v₀`. The right RoPE probe is
"RoPE vs identity rotation differ", which returns True.

## Verification of Prof. feedback

- The correlation between the reference loop and the implementation is exact to float32
  precision, so scaling, mask, softmax axis and head merge are all consistent.
- The causality test (T1) covers the mask specifically; see `notes/TESTS.md`.
