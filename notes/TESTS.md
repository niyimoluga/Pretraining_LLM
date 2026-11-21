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

## T2 — next-token alignment (to be added with TODO 4/5)

**Checks.** `make_targets([B, T+1])` returns `inputs = windows[:, :-1]` and
`targets = windows[:, 1:]`, matching a hard-coded expected pair, on the same device
with integer dtype; and the loss from `GPT.forward(ids, targets)` equals a manually
computed shifted cross-entropy.

**Expected result.** For `[[1,2,3,4]]`, `inputs = [[1,2,3]]`, `targets = [[2,3,4]]` —
each position predicts the *following* token.

**Bug it catches.** Predicting the current token instead of the next (no shift, or a
double shift), or swapping inputs and targets.

**Would miss.** Nothing about masking or clipping.

---

## T3 — clipping and ignored targets (to be added with TODO 5)

**Checks.** (a) For a loss large enough that the gradient norm exceeds 1, the returned
`grad_norm` is the norm **before** clipping (`> 1`), and the applied update equals
`lr * grad / grad_norm` applied to a cloned copy. (b) Appending `-100` to targets does
not change the loss.

**Expected result.** Pre-clip norm returned verbatim; the update direction is the
raw gradient scaled by `1/norm`; `-100` positions contribute nothing.

**Bug it catches.** Returning the post-clip norm, forgetting `clip_grad_norm_`, not
clearing old gradients, or dropping `ignore_index=-100`.

**Would miss.** Attention/block wiring.
