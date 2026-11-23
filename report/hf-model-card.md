---
language: en
library_name: pytorch
tags:
  - causal-lm
  - gpt
  - from-scratch
  - educational
  - bits-per-byte
license: mit
---

# Small decoder-only language model (Assignment 1)

Two from-scratch decoder-only Transformer language models, pre-trained on ~500M tokens of
English web text for **Assignment 1 of the *Fundamentals of LLMs* course** (University of Graz).
Both models are small (≈27.8M parameters), share an identical architecture and training setup,
and differ in **exactly one feature**: the activation in the gated feed-forward layer.

| Subdirectory | Change vs baseline | Activation | Best validation bits per byte |
|---|---|---|---|
| `baseline/` | — (course default) | SiLU | **1.3453** |
| `variant/` | gated FFN activation | GELU | **1.3509** |

Each subdirectory contains the model checkpoint, tokenizer, training logs, settings and the
exact Python source used to produce it.

## Author

- **Olaniyi Moluga** — matriculation number 12340366
- Course: **Fundamentals of LLMs**, University of Graz
- Repository: <https://github.com/niyimoluga/Pretraining_LLM>

## Architecture

A pre-norm, RoPE-based decoder-only Transformer, in the style of GPT / nanoGPT.

| Component | Setting |
|---|---|
| Layers | 8 |
| Model width (`d_model`) | 512 |
| Attention heads | 8 (head width 64) |
| Context length | 512 tokens |
| Vocabulary | 4,096 (byte-level BPE) |
| Parameters | **27,795,968** (both models) |
| FFN | gated / SwiGLU-style, hidden width 1,408 |
| Positional encoding | Rotary (RoPE) on Q and K only |
| Normalization | RMSNorm (pre-norm, both sublayers) |
| Output projection | **tied** to the input embedding matrix |
| Dropout | none |

Block computation:

```text
x = x + attention(attn_norm(x))
h = ffn_norm(x)
x = x + down(activation(gate(h)) * up(h))
```

`activation` is **SiLU** in `baseline/` and **GELU** in `variant/` — the single changed feature.

## Training

| Setting | Value |
|---|---|
| Dataset | [`idealab-cs2/fundamentals-of-llms-training-data`](https://huggingface.co/datasets/idealab-cs2/fundamentals-of-llms-training-data) (English FineWeb subset, ~1.5 GiB), revision `9624da98…` |
| Documents | 541,041 total; split at **val fraction 0.01, seed 0** → **535,601 train / 5,440 validation** (whole documents, disjoint) |
| Tokenizer | one byte-level BPE, vocab **4,096**, `<\|endoftext\|>` id 0, shared by both runs |
| Encoded tokens | **456,968,855** train / 4,507,188 validation |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 (2-D params only) |
| Learning rate | peak 1e-3, 500-step warmup, cosine decay to 10 % |
| Batch size | 32 sequences × 512 tokens = 16,384 tokens/step |
| Steps | **30,518** |
| Training budget | **500,006,912 tokens** (≈1.09 passes over the training split) |
| Precision | float32 |
| Gradient clipping | global norm 1.0 |
| Seed | 0 (identical initial weights for both runs) |
| Hardware | 1× NVIDIA **A100-PCIE-40GB** per run, ~3.2 h each, run in parallel |

The default course hardware is an RTX 2080 Ti (IT010146); that queue was fully occupied, so
both runs used the same A100 node instead. Same data, seed, schedule and token budget for both,
so the comparison remains single-variable. Grading uses bits per byte, which is hardware-independent.

## Results

Validation loss and **bits per byte** (bpb), the course's tokenizer-independent metric:

| Step | Tokens | baseline val loss | baseline bpb | variant val loss | variant bpb |
|---|---|---|---|---|---|
| 0 | 0 | 8.4175 | 3.4789 | 8.4181 | 3.4791 |
| 6,000 | 98,304,000 | 3.6892 | 1.5247 | 3.6984 | 1.5285 |
| 12,000 | 196,608,000 | 3.5192 | 1.4545 | 3.5292 | 1.4586 |
| 18,000 | 294,912,000 | 3.4046 | 1.4071 | 3.4147 | 1.4113 |
| 24,000 | 393,216,000 | 3.3091 | 1.3676 | 3.3223 | 1.3731 |
| 30,518 | 500,006,912 | **3.2551** | **1.3453** | **3.2686** | **1.3509** |

- Step-0 losses agree to four decimals, confirming both runs start from identical weights.
- Replacing SiLU with GELU changes the best validation bpb by **+0.0056 (+0.42 %)** — a near-null
  ablation, as expected when swapping one smooth activation for another in a gated FFN.
- For reference, a model guessing uniformly over 4,096 tokens scores ≈2.70 bpb on this data
  (mean 4.448 UTF-8 bytes per token), so both models learn well below the chance floor.

## Files

```
baseline/
  best.pt          checkpoint with the lowest validation loss (the reported model)
  tokenizer.json   the 4,096-token BPE tokenizer
  metrics.jsonl    per-step and validation metrics
  run.json         model, training and environment settings
  source/          exact Python source used for the run (model.py, train.py, tools.py, helpers/)
  data-meta.json   dataset counts and checksums
  environment.txt  package versions
variant/           same layout, with the GELU activation
manifest.json      package checksums
tests/             the three student tests
```

## Usage

These are custom PyTorch models, so use the **saved source** in each subdirectory rather than a
standard Hugging Face class. Install PyTorch and the tokenizer dependency (`tokenizers==0.22.2`),
then:

```python
import torch
import sys
sys.path.insert(0, "baseline/source")
from model import Config, GPT

ckpt = torch.load("baseline/best.pt", map_location="cpu", weights_only=True)
model = GPT(Config(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()

ids = torch.tensor([[ckpt["eot_id"]]])          # start from the end-of-text token
out = model.generate(ids[0].tolist(), count=40, temperature=0.8, top_k=40)
print(out)
```

The package also ships command-line helpers:

```sh
python baseline/source/tools.py generate baseline/best.pt --prompt "The purpose of a library is" --tokens 40 --device cpu
python baseline/source/tools.py evaluate baseline/best.pt --text eval.txt --device cpu
```

`evaluate` reports `bits_per_byte` using the course protocol (`packed-nonoverlapping-v1`):

```text
bits per byte = (total cross-entropy in nats) / (number of UTF-8 text bytes) / ln(2)
```

## Intended use and limitations

- **Intended for** coursework, evaluation and research on small-model pre-training. Not for
  production, factual question answering, or any high-stakes use.
- The models are trained on ~500M tokens only (~1 pass over a 457M-token corpus); generated text
  is grammatically plausible but often incoherent, and factual accuracy is not expected.
- English web text only. No instruction tuning, no RLHF, no safety alignment; outputs may
  reproduce biases or harmful patterns present in the training data.
- Maximum context is 512 tokens. The tokenizer is fixed at 4,096 tokens and must match the
  checkpoint.

## Acknowledgements

Assignment and evaluation protocol by the *Fundamentals of LLMs* course, University of Graz
(instructors Lorenz Prattes and Ruggero Lazzaroni). Architecture inspired by Karpathy's
nanoGPT and the CS336 component-based approach. Training data from the course
`idealab-cs2/fundamentals-of-llms-training-data` dataset.
