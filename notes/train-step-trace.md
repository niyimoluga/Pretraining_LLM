# TODO 5 — `train_step`: contract, verification and the non-contiguous bug

Implemented 2026-11-20 in `assignment1/code/train.py`.

## What it does

```python
model.train()
optimizer.zero_grad(set_to_none=True)
with precision_context:
    _, loss = model(x, y)
if not torch.isfinite(loss): raise FloatingPointError(...)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # returns PRE-clip norm
if not torch.isfinite(grad_norm): raise FloatingPointError(...)
optimizer.step()
return float(loss.detach()), float(grad_norm)
```

## Why each step (defence points)

- **`model.train()`** — puts any mode-dependent layer into training behaviour. There is no
  dropout in this model, but the call is correct and required by the spec.
- **`zero_grad(set_to_none=True)`** — clears stale gradients. `set_to_none=True` frees the
  memory and is the modern default; without clearing, gradients would *accumulate* across
  steps and the second update would be wrong.
- **`with precision_context`** — the caller supplies `nullcontext()` on CPU or autocast on
  capable CUDA. Keeping the forward inside it means mixed precision covers the whole graph.
- **Finite-loss check before `backward()`** — a NaN/Inf loss would produce NaN gradients for
  every parameter, silently destroying the model. Fail fast instead.
- **`clip_grad_norm_(params, 1.0)`** — scales all gradients by `1/norm` when the total norm
  exceeds 1.0. Its **return value is the norm before clipping**, which is exactly what the
  trainer logs as `grad_norm` and the plots annotate ("norms above 1.0 mean clipping was
  applied"). Returning the post-clip norm would be a subtle but wrong log.
- **Finite-gradient check before `step()`** — never move weights with a NaN gradient.
- **Do not set the learning rate** — the supplied loop sets `group["lr"]` from the warmup +
  cosine schedule before calling `train_step`.
- **Return Python floats** — the caller validates `isinstance(..., (float, int))` and
  `math.isfinite`.

## Verification (tests + required fixed-batch check)

### Student test T3

`test_train_step_clips_gradients_and_clears_grads` uses two toy models with a known gradient:

| Regime | Expected | Result |
|---|---|---|
| grad norm 100 (>> 1) | returned `grad_norm == 100` (pre-clip); SGD update `w = -lr` (unit direction), not `-lr*100` | ✅ |
| grad norm 0.001 (< 1) | no clipping; two identical steps give the same delta (grads cleared) | ✅ |

`test_train_step_ignores_masked_targets` checks the returned loss equals a manual
`F.cross_entropy(..., ignore_index=-100)` on a `-100`-padded batch. ✅

### Required "learn one fixed batch" check (`notes/overfit-check.txt`)

Config: 1 layer, width 16, 2 heads, context 8, two fixed 9-token sequences, 100 AdamW
updates (lr 3e-3) on the *same* batch.

| Step | loss | grad_norm (pre-clip) |
|---|---|---|
| 1 | 3.49079 | 1.869 |
| 10 | 2.91587 | 1.035 |
| 25 | 2.16788 | 1.003 |
| 50 | 1.21253 | 0.858 |
| 75 | 0.51888 | 0.671 |
| 100 | 0.18618 | 0.314 |

Loss falls **3.49 → 0.186** — the model can clearly fit a fixed batch, so the whole
forward/backward/update path is correct. The norm starts above 1.0 (clipping active early)
and drops below it as the fit improves, which is the expected pattern.

## The bug the supplied tests caught (worth keeping for the report)

The first version of `GPT.forward` used `targets.view(-1)`. That works for contiguous
targets but **raises** for the non-contiguous slice that `make_targets` returns
(`windows[:, 1:]`):

```text
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

My own T3 tests happened to pass contiguous targets, so they did **not** catch it — the
supplied `test_training_step_runs` did, because it goes through `make_targets`. Fixed by
switching to `.reshape(-1)` (which copies only when necessary). This is a good example of
why the supplied end-to-end test exists alongside the focused ones.

After the fix: `python -m pytest assignment1/tests -q` → **8 passed**.