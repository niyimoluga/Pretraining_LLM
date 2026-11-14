# What every file in `assignment1/code` is for

You only edit **three** files: `model.py`, `train.py`, `tests/test_student.py`.
Everything else is supplied infrastructure — read it, don't change it.

## Files you edit

| File | Role | What you write |
|---|---|---|
| `model.py` | the architecture (`Config`, `Attention`, `Block`, `GPT`) | `Attention.forward` (TODO 1), `Block.forward` (TODO 2), `GPT.forward` (TODO 3). Layer constructors, RoPE tables, init and `generate()` are supplied |
| `train.py` | objective + one update | `make_targets` (TODO 4), `train_step` (TODO 5). Its `__main__` just calls `run_training(parse_args(), ...)` |
| `tests/test_student.py` | your 3 tests | causal-mask invariance, target alignment, clipping/`-100` handling (+ bug demo) |

## Supplied code next to them

| File | Role |
|---|---|
| `components.py` | `RMSNorm` (fp32 accumulation) and `rotate` (RoPE on even/odd pairs). 19 lines — read both before writing attention |
| `tools.py` | tiny CLI dispatcher: `python assignment1/code/tools.py {split-data,train-tokenizer,prepare,tokenize,generate,evaluate,plot-runs,submission} …` → imports the matching `helpers/*.py` and runs its `main()` |
| `helpers/bpe.py` | the course's Rust byte-level BPE backend (via the `tokenizers` package), the `<|endoftext|>` id, and the reversible byte alphabet used for bytes-per-byte counting |
| `helpers/tokenizer.py` | `Tokenizer` wrapper: `encode`, `decode`, `token_bytes(id)` (needed for bpb), `from_file/from_str`, `to_json`. This is why bpb can be tokenizer-independent |
| `helpers/data.py` | all data stages: `split-data` (whole documents, seed, `split.json` + `laptop-*` subsets, checksums, overlap check), `train-tokenizer` (samples 32 MiB, vocab 4096), `prepare` (encode → `train.bin`/`val.bin` uint32 + `meta.json`), `tokenize` (debug a sentence) |
| `helpers/infrastructure.py` | the trainer: `load_data` (sha256 + memmap), `batch` (random windows), `make_targets` is *your* function it calls, `learning_rate` (warmup + cosine to 10%), `autocast`, `train_step` is *your* function, `evaluate` (non-overlapping packed windows, counts end-token loss but not end-token bytes), checkpoint saving, `run.json`, copies `source/`, argparse with JSON-config defaults |
| `helpers/tracking.py` | optional Weights & Biases logging — no-op unless `--wandb-project` is given |
| `helpers/evaluate.py` | CLI `evaluate`: scores any checkpoint on an arbitrary UTF-8 file with the official `packed-nonoverlapping-v1` protocol → nats/token and bits per byte. This is what the teaching team's held-out evaluation mirrors |
| `helpers/generate.py` | CLI `generate`: samples text from a checkpoint using the tokenizer embedded in it |
| `helpers/plot_runs.py` | CLI `plot-runs`: reads two runs' `metrics.jsonl`, writes loss/val-loss/bpb/grad-norm panels (`comparison.png` for the report) |
| `helpers/submission.py` | CLI `submission {package,upload,check}`: validates both runs (steps, tokens, hashes), builds `baseline/` + `variant/` with logs/settings/source/tests, creates the HF repo, uploads, writes `a1-receipt.json` (repo link + revision), then verifies the download |
| `helpers/__init__.py` | package marker |

## Non-code files

| Path | Role |
|---|---|
| `configs/laptop.json` | 2 layers / 128 width / 4 heads / ctx 128 / 500 steps, lr 3e-3 — your dev loop |
| `configs/final.json` | 8 layers / 512 width / 8 heads / ctx 512 / batch 32 / 30,518 steps = 500,006,912 tokens, lr 1e-3 |
| `requirements.txt` | numpy, pytest, tokenizers==0.22.2, huggingface_hub==1.33.0, matplotlib (torch installed separately) |
| `tests/test_interfaces.py` | supplied smoke tests (shapes, finite loss, one step runs). Deliberately does **not** check masking, alignment or updates |
| `tests/test_bpe_interfaces.py` | supplied tokenizer/round-trip checks |
| `../train.sbatch` | Slurm job: 1× RTX 2080 Ti on `IT010146`, fp32, 6 h. Runs `.venv/bin/python assignment1/code/train.py --data $1 --out $2 --config final.json --device cuda --precision float32` plus any extra flags (e.g. `--activation gelu`) |
| `../prepare.sbatch` | Slurm CPU job (`hpc` partition) that runs split → tokenizer → prepare (laptop + final) on the cluster, for when the laptop is too slow to encode 1.5 GiB |
| `../assignment.md`, `../submission.md` | the spec and the hand-in procedure |
| `../practicals/setup.md`, `gpu-check.sbatch` | environment + GSC1 setup, and the 5-minute GPU smoke job |

## Who calls whom (training path)

```text
train.py __main__
  └─ helpers.infrastructure.run_training(args, make_targets, train_step)
       ├─ load_data(data/a1/final)      → memmap train/val + tokenizer + byte lengths
       ├─ GPT(Config(...))              → model.py (uses components.py)
       ├─ AdamW / learning_rate         → supplied
       ├─ batch(...) → make_targets     → your TODO 4
       ├─ train_step(...)               → your TODO 5 (forward/backward/clip/step)
       ├─ evaluate(...)                 → GPT.forward → val loss + bits per byte
       └─ saves best.pt / last.pt / metrics.jsonl / run.json / source/
```

## Why the `source/` copy matters

Each run directory contains a snapshot of the code that produced it, plus
`run.json` with hashes, config, device and library versions. The submission script
re-verifies these and the packaged tests are re-run **against the variant's saved
source** — so a run made with stale code, or a checkpoint whose code later changed,
is detected. Practical consequence: freeze `model.py` / `train.py` before the two
GPU runs, and never "quickly edit" them afterwards.
