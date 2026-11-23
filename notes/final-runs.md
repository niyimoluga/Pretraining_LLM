# Final GPU runs — results (2026-11-23)

Assignment 1 requires **two runs of ~500M tokens each** on one GPU: a baseline and one
modified model, with everything else held fixed.

## Runs

| Run | Job | Node / GPU | Activation | Time limit | Runtime | W&B |
|---|---|---|---|---|---|---|
| `runs/a1-baseline` | 5836749 | IT010213 / A100-PCIE-40GB | SiLU | 12 h | **3.18 h** | `2ogs1uym` |
| `runs/a1-variant` | 5836750 | IT010213 / A100-PCIE-40GB | GELU (`--activation gelu`) | 12 h | **3.17 h** | `gfp1jco6` |

Both: `configs/final.json` (8 layers, width 512, 8 heads, context 512, batch 32), fp32,
30,518 steps = 500,006,912 tokens, seed 0, same data (`data/a1/final`) and tokenizer.
Both `COMPLETED` with exit code `0:0` and ended at exactly `step: 30518`,
`tokens: 500006912`.

## Results

| Step | Tokens | baseline val_loss | baseline bpb | variant val_loss | variant bpb |
|---|---|---|---|---|---|
| 0 | 0 | 8.4175 | 3.4789 | 8.4181 | 3.4791 |
| 6,000 | 98,304,000 | 3.6892 | 1.5247 | 3.6984 | 1.5285 |
| 12,000 | 196,608,000 | 3.5192 | 1.4545 | 3.5292 | 1.4586 |
| 18,000 | 294,912,000 | 3.4046 | 1.4071 | 3.4147 | 1.4113 |
| 24,000 | 393,216,000 | 3.3091 | 1.3676 | 3.3223 | 1.3731 |
| 30,000 | 491,520,000 | 3.2570 | 1.3461 | 3.2704 | 1.3516 |
| **30,518** | **500,006,912** | **3.2551** | **1.3453** | 3.2686 | 1.3509 |

**Best validation bits per byte (lower is better):**

| Model | Best val bpb | Final train loss | Final grad norm |
|---|---|---|---|
| baseline (SiLU) | **1.3453** | 3.2976 | 0.528 |
| variant (GELU) | **1.3509** | 3.3090 | 0.519 |

Difference: **+0.0056 bpb (+0.42 %) for GELU** — i.e. the baseline is very slightly better.

## Interpretation

- **Step-0 losses match to 4 decimals** (3.4789 vs 3.4791) → identical initial weights; the only
  difference between the runs is the activation function. Clean single-variable ablation.
- The two curves are essentially coincident for the whole 500M-token budget
  (`report/comparison.png`), with GELU consistently a hair *worse*. With SwiGLU-style gating,
  SiLU and GELU are both smooth ReLU-family shapes, so this is the expected near-null result;
  the small gap is within run-to-run noise at this scale and should not be over-interpreted.
- Convergence is healthy: val bpb 3.479 → 1.345, i.e. far below the uniform-model floor of
  ≈2.70 bpb, and still improving slowly at the end (no overfitting; ~1.09 passes over the data).
- Gradient norm spikes above 1.0 only in the first hundreds of steps (clipping active), then
  settles around 0.5 — visible in the 4th panel of the figure.

## Where the models are saved on GSC1

```
/cl_tmp/23moluga/llm-course/llm-assignment1/runs/
├── a1-baseline/
│   ├── best.pt            107 MB  ← lowest validation loss checkpoint (this is the submission)
│   ├── last.pt            107 MB  ← final-step checkpoint
│   ├── metrics.jsonl              one JSON row per step + validation
│   ├── run.json                   model/training/environment settings
│   ├── source/                    exact code snapshot used for the run
│   ├── tokenizer.json             tokenizer used
│   ├── data-meta.json             dataset counts + checksums
│   ├── environment.txt            pip freeze
│   └── wandb/                     local W&B files
└── a1-variant/            same layout, activation = gelu
```

Runs on `/cl_tmp` are **not backed up**: the models are copied to Hugging Face and the figures/
notes to the GitHub repo, which is the durable copy.

## Deviation recorded for the report

The course default is one **RTX 2080 Ti on IT010146**. All 8 cards there were occupied by other
users (one holding 7 for 1–5 days) with `StartTime=Unknown`, so both runs were executed on
**A100-PCIE-40GB (IT010213)** instead, in parallel. Same hardware for both runs, same data,
seed and token budget; grading is bits-per-byte on held-out text, which is hardware-independent.
Only the reported wall time reflects the faster card. `train.sbatch` was also changed from
`--time=06:00:00` to `--time=12:00:00` (the final runs needed 3.2 h).
