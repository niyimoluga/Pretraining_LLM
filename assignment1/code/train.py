"""Implement target construction and one optimizer update, then run the trainer."""
import torch

from helpers.infrastructure import parse_args, run_training


def make_targets(windows):
    """TODO 4: split integer windows [B, T+1] into inputs and next-token targets.

    Return two integer tensors [B, T] on the same device, preserving the
    original token order. See assignment.md for the next-token objective.
    """
    # Slice views, so dtype, device, memory order and gradient-relevant layout
    # are preserved without a copy. Each row of windows has T+1 tokens; the
    # first T are inputs and the last T are the next-token targets, i.e. the
    # inputs shifted one position to the left. Position i therefore predicts
    # window[i + 1]. Do not shift anywhere else — GPT.forward assumes this.
    return windows[:, :-1], windows[:, 1:]


def train_step(model, optimizer, x, y, precision_context):
    """TODO 5: perform one optimizer update; return (loss, grad_norm) as floats.

    Enter training mode and clear old gradients. Within precision_context,
    run the model and obtain its loss. Reject a nonfinite loss. Backpropagate,
    clip the total gradient norm to 1.0 (reject nonfinite gradients), then
    update the parameters. grad_norm is the value returned by clip_grad_norm_:
    the total norm before clipping. The supplied caller sets the learning rate.
    precision_context is a context manager, including on CPU.
    """
    # Training mode and cleared gradients: do not accumulate onto a previous
    # update, and let any mode-dependent layer behave as during training.
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Forward pass under the caller's precision context (no-op on CPU, autocast on
    # capable CUDA). The loss is the mean next-token cross-entropy.
    with precision_context:
        _, loss = model(x, y)

    # A nonfinite loss would poison every parameter through the gradients.
    if not torch.isfinite(loss):
        raise FloatingPointError(f"nonfinite loss: {loss.detach().item()}")

    # Backpropagation fills .grad on every parameter that requires one.
    loss.backward()

    # Clip the *total* gradient norm to 1.0. The returned value is the norm BEFORE
    # clipping (the trainer logs and plots it). Reject nonfinite gradients before
    # they can reach the weights.
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"nonfinite gradient norm: {grad_norm.detach().item()}")

    # Apply the update. The caller already set the learning rate on the optimizer,
    # so we must not change it here.
    optimizer.step()

    # Two Python floats: the pre-clip loss and the pre-clip gradient norm.
    return float(loss.detach()), float(grad_norm)


if __name__ == "__main__":
    run_training(parse_args(), make_targets, train_step)
