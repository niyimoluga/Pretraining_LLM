# Student tests and bug demonstration

File: `assignment1/tests/test_student.py`. Run from the project root:

```sh
python -m pytest assignment1/tests -q
```

Each test below states **what it checks**, **the bug it catches**, and **what it
would miss**. Three tests are required by the assignment; they are added as their
code is implemented.

---

## T1 — `test_attention_is_causal` (implemented)

**Checks.** `Attention.forward` is causal: for a sequence of length 8, replacing
tokens at positions 4–7 with different values leaves the outputs at positions 0–3
unchanged (`allclose`, atol 1e-6), while positions 4–7 are free to change. It
targets the `Attention` module directly, building the RoPE tables itself so the
test does not depend on `GPT.forward`.

**Why the expected result is what it is.** Causal attention lets query position `i`
read only keys `j <= i`. Pulling the lower-triangular mask to the strict diagonal
means positions 0–3 physically cannot see positions 4–7, so their outputs are
deterministic functions of the first four tokens alone.

**Bug it catches.** A wrong mask: removed entirely, inverted (`triu` instead of
`tril`), or applied to the wrong axis. The demonstration below injects exactly this.

**What it would miss.** Errors that preserve causality but corrupt values: wrong
scaling (`sqrt(C)` instead of `sqrt(D)`), a softmax over the query axis instead of
the key axis, RoPE applied to V as well as Q/K, or heads merged in the wrong order.
Those need the reference-comparison check in `notes/attention-trace.md`.

### Bug demonstration (required deliverable)

Patch: `notes/bug-demo.patch` — flips the mask from lower- to upper-triangular:

```diff
-        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
+        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device))  # BUG
```

| Run | Command | Output | Result |
|---|---|---|---|
| Bug present | `pytest assignment1/tests/test_student.py -v` | `notes/bug-demo-fail.txt` | **1 failed** — `assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6)` → `assert False` |
| Bug fixed | same | `notes/bug-demo-pass.txt` | **1 passed** |

The failure is the first assertion: with `triu`, position 0 attends to all
positions, so perturbing tokens 4–7 changes its output and the "past is unchanged"
property breaks. Restoring `tril` makes it pass.

---

## T2 — `test_make_targets_predicts_next_token` (implemented)

**Checks.** `make_targets([[1,2,3,4],[5,6,7,8]])` returns exactly
`inputs = [[1,2,3],[5,6,7]]` and `targets = [[2,3,4],[6,7,8]]`; the structural shift
relation `inputs[:, 1:] == targets[:, :-1]`; and that shape, dtype and device are
preserved. Slice views mean these are held exactly, not approximately.

**Why the expected result is what it is.** A window of `T+1` tokens is consumed as `T`
prediction problems: the first `T` tokens are the inputs, and the token after each one
is its target. So for `1,2,3,4` position 0 (input 1) predicts 2, position 1 predicts 3,
position 2 predicts 4 — the targets are the inputs shifted one left.

**Bug it catches.** Predicting the current token (no shift: `targets == inputs`), a
double shift (predicting two ahead), swapped inputs and targets, or reversing the token
order inside each row. All of these are shape- and dtype-correct, so the supplied
`test_target_shapes` does not detect them.

**Would miss.** Causal masking (T1), gradient clipping and optimiser behaviour (T3).

---

## T3 — `test_train_step_clips_gradients_and_clears_grads` (implemented)
plus `test_train_step_ignores_masked_targets`

**Checks.** With a toy model whose gradient norm is exactly 100, `train_step` returns the
**pre-clip** norm (100) and applies an SGD update of `-lr` (the clipped unit direction),
not `-lr*100`. With a gradient norm of 0.001 (< 1), no clipping occurs and two identical
steps make the same parameter change — proving old gradients are cleared. A second test
checks the returned loss equals a manual `F.cross_entropy(..., ignore_index=-100)` on a
`-100`-padded batch.

**Why the expected results are what they are.** `clip_grad_norm_` rescales by `1/norm`
only when `norm > 1`; because the second regime has norm 0.001, the step is proportional
to the gradient, so an uncleared (accumulated) gradient would double the second delta.

**Bug it catches.** Returning the post-clip norm, forgetting to clip, clipping after the
optimizer step, never clearing gradients (accumulation), or dropping `ignore_index`.

**Would miss.** Attention masking (T1) and target alignment (T2).
