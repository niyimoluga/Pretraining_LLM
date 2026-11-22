"""Student tests for Assignment 1. See ../assignment.md.

Each test checks behaviour the supplied smoke tests do not. Explanations (what
each test catches and what it would miss) are in notes/TESTS.md.

Run from the project root:
    python -m pytest assignment1/tests -q
"""
from pathlib import Path
import sys
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from model import Attention, Config, GPT
from train import make_targets, train_step


def _rope_tables(context, head_dim):
    """Same rotary tables GPT builds, exposed here so Attention can be tested alone."""
    angles = torch.outer(
        torch.arange(context).float(),
        10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim),
    )
    return angles.cos(), angles.sin()


def test_attention_is_causal():
    """T1: changing tokens after position t must not change outputs at positions <= t.

    Causal attention lets position i read only positions j <= i. So if we replace
    everything from position 4 onward with different values, the outputs at
    positions 0..3 must be bit-for-bit identical, while positions 4..7 are free
    to change.

    Catches: a removed or inverted mask (tril vs triu), or masking the wrong axis.
    Would miss: wrong scaling, wrong head merging, or a softmax over the wrong axis
    (those keep causality but change values).
    """
    config = Config(vocab_size=32, context=8, layers=1, heads=2, width=16)
    attn = Attention(config)
    attn.eval()
    cos, sin = _rope_tables(config.context, config.width // config.heads)

    torch.manual_seed(0)
    x = torch.randn(2, config.context, config.width)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])  # perturb the "future"

    with torch.no_grad():
        before = attn(x, cos, sin)
        after = attn(changed, cos, sin)

    assert before.shape == (2, config.context, config.width)
    # Positions 0..3 cannot see changes at 4..7 -> identical.
    assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6)
    # The mask must not block the legitimate dependencies: later positions change.
    assert not torch.allclose(before[:, 4:], after[:, 4:], atol=1e-4)


def test_make_targets_predicts_next_token():
    """T2: every position predicts the FOLLOWING token.

    ``make_targets`` receives [B, T+1] windows and must return [B, T] inputs and
    [B, T] targets where position i predicts window[i + 1]. The supplied tests only
    check shapes and dtype, so a model that predicts the *current* token (no shift)
    or the token two ahead (double shift) would still pass them.

    The check is exact: a hard-coded window with distinct values, plus the shift
    relation targets[:, :-1] == inputs[:, 1:].

    Catches: no shift (targets == inputs), a double shift, swapped inputs/targets,
    or reversed token order.
    Would miss: causal masking, clipping, or any optimiser behaviour.
    """
    windows = torch.tensor([[1, 2, 3, 4],
                            [5, 6, 7, 8]])
    inputs, targets = make_targets(windows)

    expected_inputs = torch.tensor([[1, 2, 3],
                                    [5, 6, 7]])
    expected_targets = torch.tensor([[2, 3, 4],
                                     [6, 7, 8]])
    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(targets, expected_targets)

    # Structural property: dropping the first target reproduces the remaining inputs.
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    # Shape, dtype and device are preserved (slices, not rebuilt tensors).
    assert inputs.shape == targets.shape == (2, 3)
    assert inputs.dtype == targets.dtype == windows.dtype
    assert inputs.device == targets.device == windows.device


class _ToyLM(torch.nn.Module):
    """Minimal model exposing the GPT interface with a controllable gradient.

    forward returns a loss of ``w * scale`` so the gradient with respect to ``w``
    is exactly ``scale`` and its norm is easy to predict.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.w = torch.nn.Parameter(torch.zeros(()))

    def forward(self, ids, targets=None):
        return ids, self.w * self.scale


def test_train_step_clips_gradients_and_clears_grads():
    """T3: clipping at norm 1.0, the PRE-clip norm is returned, and grads are cleared.

    Catches: returning the post-clip norm, skipping clip_grad_norm_, clipping after
    the optimizer step, or never zeroing gradients between updates.
    Would miss: wrong attention/block wiring, or wrong target alignment (T1/T2).

    Two regimes pin the contract down exactly:
    (a) gradient norm 100 >> 1: clipping must turn the step into -lr * (unit direction);
    (b) gradient norm 0.001 < 1: no clipping, and two identical steps must make the
        same change (proving old gradients are cleared).
    """
    x = torch.zeros(1, 4, dtype=torch.long)
    y = torch.zeros(1, 4, dtype=torch.long)

    # (a) clipping engages: grad norm is 100, so the update magnitude is lr, not lr*100.
    big = _ToyLM(scale=100.0)
    opt_big = torch.optim.SGD(big.parameters(), lr=0.1)
    loss, grad_norm = train_step(big, opt_big, x, y, nullcontext())
    assert isinstance(loss, float) and isinstance(grad_norm, float)
    assert grad_norm == pytest.approx(100.0)          # norm BEFORE clipping is returned
    assert big.w.item() == pytest.approx(-0.1)        # -lr * clipped(unit) direction

    # (b) clipping does not engage, and gradient clearing is observable:
    #     with a stale gradient, the second step would move twice as far.
    small = _ToyLM(scale=0.001)
    opt_small = torch.optim.SGD(small.parameters(), lr=1.0)
    _, norm1 = train_step(small, opt_small, x, y, nullcontext())
    first_delta = small.w.item()
    _, norm2 = train_step(small, opt_small, x, y, nullcontext())
    second_delta = small.w.item() - first_delta
    assert norm1 == pytest.approx(0.001) and norm2 == pytest.approx(0.001)
    assert second_delta == pytest.approx(first_delta)  # grads were cleared, not summed


def test_train_step_ignores_masked_targets():
    """T3 (continued): the loss returned by train_step ignores target == -100.

    Catches: a missing/incorrect ignore_index, or dropping masking before the loss.
    Would miss: masking in attention (T1) and clipping (previous test).
    """
    cfg = Config(vocab_size=32, context=8, layers=1, heads=2, width=16)
    torch.manual_seed(0)
    reference = GPT(cfg)
    trained = GPT(cfg)
    trained.load_state_dict(reference.state_dict())   # same starting weights

    ids = torch.randint(0, 32, (1, 8))
    targets = ids.clone()
    targets[:, 4:] = -100                             # only the first four are scored

    with torch.no_grad():
        logits, _ = reference(ids)
        manual = F.cross_entropy(logits.float().view(-1, cfg.vocab_size),
                                 targets.view(-1), ignore_index=-100)

    optimizer = torch.optim.SGD(trained.parameters(), lr=0.0)  # no parameter movement
    loss, _ = train_step(trained, optimizer, ids, targets, nullcontext())
    assert loss == pytest.approx(manual.item(), rel=1e-6)
