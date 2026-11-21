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
