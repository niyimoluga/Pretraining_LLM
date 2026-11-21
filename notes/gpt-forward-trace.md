# TODO 3 — `GPT.forward`: structure, verification and defence notes

Implemented 2026-11-20 in `assignment1/code/model.py`.

## Data flow

```python
assert ids is [B, T] and 1 <= T <= context        # else ValueError
x = self.embedding(ids)                            # [B, T, C]
for block in self.blocks: x = block(x, cos, sin)   # [B, T, C]
x = self.norm(x)                                   # final RMSNorm
logits = x @ self.embedding.weight.T               # tied unembedding -> [B, T, vocab]
loss = cross_entropy(logits.float().view(-1, V), targets.view(-1), ignore_index=-100)
return logits, (loss or None)
```

## Why each choice (defence points)

- **Input validation up front.** `ids.ndim != 2` or `T` outside `[1, context]` raises
  `ValueError`. The position tables (`cos`, `sin`) only cover `context` rows, and the
  attention builder slices `cos[:T]`, so an out-of-range `T` would silently mis-index —
  we reject it instead.
- **Tied embeddings.** There is no separate output layer: `self.embedding.weight` is reused
  as the unembedding and transposed, so `logits = x @ Wᵀ` with `W = [vocab, C]`. This saves
  a `vocab × C` matrix (2,097,152 params in the final model) and is the standard GPT trick.
  It also means gradients from the loss flow into the *same* matrix used for the input
  lookup — verified below.
- **Final RMSNorm before unembedding.** Keeps the logit scale controlled regardless of
  residual accumulation through the stack.
- **Loss in float32.** `logits.float()` before cross-entropy, matching the evaluator and
  the fp32-accumulation convention used by RMSNorm and the softmax.
- **Targets are already shifted.** The caller (`make_targets`) produces `inputs = w[:-1]`,
  `targets = w[1:]`. `GPT.forward` must not shift again — shifting twice is the classic
  "predicts two ahead / one behind" bug.
- **`ignore_index=-100`.** Positions with target `-100` contribute nothing to the mean.
  This is what makes the evaluator's right-padding safe.

## Verification (scratch check, all reproduced)

| Check | Result |
|---|---|
| logits shape `[B,T,vocab]` | `(2, 8, 32)` ✅ |
| loss finite; `targets=None` → `loss is None` | ✅ |
| loss equals a manual `F.cross_entropy` on the given (already shifted) targets | **True** |
| tied projection: `logits == norm-out @ embedding.weight.T` exactly | **True** |
| `-100` masking equals a manual CE over only the valid targets | **True** |
| **padding invariance**: same loss for length 5, and for length 8 with the last 3 targets `-100` | **True** (3.118218 both) |
| gradients reach `embedding.weight` from the loss (tied in+out) | **True** |
| rejects 1-D input, `T > context`, and `T = 0` with `ValueError` | **True** |
| `T == context` works (boundary) | **True** |
| all targets `-100` ⇒ loss is `nan` (surfaces rather than silently 0) | **True** |

### Why padding invariance matters

The supplied `evaluate` packs the validation stream into fixed-length windows and
right-pads the final partial window with `-100` targets. Because attention is causal, the
padding tokens cannot influence earlier positions, and `ignore_index` keeps the padding
out of the mean — so a padded short sequence scores identically to the unpadded one. The
check above confirms this to float32 precision.

## Supplied-test status after this TODO

`python -m pytest assignment1/tests -q` → **3 passed, 2 failed**. The two failures are the
still-unimplemented `make_targets` and `train_step` (TODOs 4–5); the forward-shape and
finite-loss supplied check now passes.
