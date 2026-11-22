# Laptop runs on the NVIDIA T500 (4 GB) — 2026-11-22

Purpose: confirm the full pipeline trains on real data, check VRAM headroom, and try the
GELU modification before the GPU runs. Both runs use `assignment1/configs/laptop.json`
(2 layers, width 128, 4 heads, context 128, batch 8, 500 steps, lr 3e-3, seed 0) with
`--device cuda --precision float32`.

## Environment / VRAM

| Item | Value |
|---|---|
| GPU | NVIDIA T500, 4096 MiB total, 0 MiB idle |
| bfloat16 natively supported | **False** (Turing) → the trainer correctly chose **float32** |
| Peak GPU memory used during training | **213 MiB** (≈5 % of 4 GB) |
| `batch_size` change needed | **No** — batch 8 × context 128 fit comfortably |
| Parameters | 950,912 |

So on a 4 GB T500 there is ~3.8 GB of headroom for the laptop config; no OOM tuning was
required. (The final config is much larger and is intended for the cluster GPU.)

## Commands

```sh
python assignment1/code/train.py --config assignment1/configs/laptop.json \
  --data data/a1/laptop --out runs/laptop-baseline --device cuda --precision float32 > runs/laptop-baseline.log 2>&1

python assignment1/code/train.py --config assignment1/configs/laptop.json \
  --data data/a1/laptop --out runs/laptop-variant --device cuda --precision float32 \
  --activation gelu > runs/laptop-variant.log 2>&1

python assignment1/code/tools.py plot-runs runs/laptop-baseline runs/laptop-variant \
  --out report/laptop-comparison.png
```

## Results

| Step | baseline val_loss | baseline bpb | variant val_loss | variant bpb |
|---|---|---|---|---|
| 0 | 8.3405 | 3.1655 | 8.3404 | 3.1655 |
| 100 | 7.6608 | 2.9076 | 7.6695 | 2.9109 |
| 200 | 7.4615 | 2.8319 | 7.2972 | 2.7696 |
| 300 | 7.0689 | 2.6829 | 7.0374 | 2.6710 |
| 400 | 6.9450 | 2.6359 | 6.8953 | 2.6170 |
| 500 | **6.8742** | **2.6090** | **6.8211** | **2.5889** |

- Wall time: baseline **12.19 s**, variant **13.35 s** of training (≈24 ms/step).
- Tokens: 512,000 each.
- Best validation = final step for both (val loss decreases monotonically).
- Max pre-clip gradient norm: baseline **1.0207** (clipping fired briefly at the start,
  then norms settled around 0.4–0.57); the plot's gradient panel shows this clearly.

## Observations

1. **The pipeline is correct end to end.** Loss falls 8.34 → 6.87 (baseline) and validation
   bits per byte 3.17 → 2.61, monotonically. Generation from `best.pt` runs and produces
   text (poor quality at 500 steps / 512k tokens, as expected; not a submission artefact).
2. **Identical initial weights.** Step-0 validation loss is 8.3405 vs 8.3404, i.e. the same
   to four decimals — confirming both runs start from the same random weights (same seed and
   construction order) and differ **only** by the activation. This is exactly the control the
   ablation needs.
3. **GELU vs SiLU on the laptop:** final bpb **2.5889 vs 2.6090**, a difference of **-0.020
   (-0.77 %)** in GELU's favour. At 500 steps / 0.5M tokens this is well inside run-to-run
   noise and should **not** be over-interpreted; it only shows the switch works and behaves
   sensibly. The real comparison happens on the 500M-token GPU runs.
4. **Sanity vs the random floor.** A uniform model would score ≈ 2.70 bpb (see
   `notes/00-phase0-1-setup.md`); these runs reach ≈2.6, already below that floor, so the
   model is genuinely learning structure rather than noise.

## Artefacts

```
runs/laptop-baseline/            metrics.jsonl, best.pt, last.pt, run.json, source/, tokenizer.json
runs/laptop-baseline.log         full stdout (one JSON line per log step)
runs/laptop-variant/             same layout, activation=gelu
runs/laptop-variant.log
report/laptop-comparison.png     4-panel figure (train loss, val loss, val bpb, grad norm)
```

`runs/` is gitignored (checkpoints are large and belong on Hugging Face); the figure and
this note are tracked.