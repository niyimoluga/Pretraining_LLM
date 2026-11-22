"""Student tests for Assignment 1. See ../assignment.md.

Each test checks behaviour the supplied smoke tests do not. Explanations (what
each test catches and what it would miss) are in notes/TESTS.md.

Run from the project root:
    python -m pytest assignment1/tests -q
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from model import Attention, Config
from train import make_targets


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
