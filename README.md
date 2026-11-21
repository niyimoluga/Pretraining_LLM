# Assignment 1: build and train a small language model

Course: *Fundamentals of LLMs* (University of Graz).
Student: Olaniyi Moluga · Hugging Face: [lanretto](https://huggingface.co/lanretto)

This repository contains the implementation, tests, logs, report and commands for
Assignment 1: a small decoder-only GPT (28M parameters, 4096-token BPE vocabulary)
pre-trained on ~500M tokens twice — a **SiLU baseline** and a **GELU variant** —
and compared in validation bits per byte.

> Status: **environment, data split, tokenizer and token files are complete; the five
> implementation TODOs, the student tests and all training runs are still pending.**
> Detailed progress report: [`STATUS.md`](STATUS.md). Models will be published at
> `lanretto/llm-assignment1` on Hugging Face (link + full revision ID go here once uploaded).

## Current status at a glance

| Phase | State |
|---|---|
| Environment (uv, venv, torch/CUDA, packages) | ✅ done |
| Dataset download (pinned revision) | ✅ done — 1.6 GB, 541,041 documents |
| Document split (1%, seed 0) | ✅ done — 535,601 train / 5,440 val |
| 4096-token BPE tokenizer | ✅ done |
| Laptop + final token files (`train.bin`/`val.bin`) | ✅ done |
| `Attention.forward` | ✅ TODO 1 (+ test T1, bug demo) |
| `Block.forward` (SiLU + GELU) | ✅ TODO 2 |
| `GPT.forward` | ⬜ TODO 3 (stub) |
| `make_targets`, `train_step` | ⬜ TODO 4–5 (stubs) |
| Three student tests + bug demonstration | ⬜ not written |
| Laptop training run, GPU baseline + variant runs | ⬜ not run (GSC1 login blocked) |
| Hugging Face upload, report, GitHub push | ⬜ not started |

**Prepared data (do not re-run unless you want to rebuild it):**

| Directory | Contents |
|---|---|
| `data/a1/tokenizer/` | `tokenizer.json` (vocab 4096, `eot_id` 0) + provenance |
| `data/a1/laptop/` | `train.bin` (600,109 tokens), `val.bin` (17,227) |
| `data/a1/final/` | `train.bin` (456,968,855 tokens), `val.bin` (4,507,188) |
| `data/a1/splits/` | `train.txt`, `val.txt`, laptop subsets, `split.json` |

> There is no `data/processed/` directory: this course stores the encoded files under
> `data/a1/`. `data/`, `runs/` and `submissions/` are git-ignored.

## Layout

```text
assignment1/
  code/model.py            causal self-attention, transformer block, GPT forward  (student)
  code/train.py            next-token targets, one optimizer update              (student)
  code/components.py       supplied RMSNorm and RoPE rotate helper
  code/tools.py            CLI: split-data | train-tokenizer | prepare | tokenize |
                           generate | evaluate | plot-runs | submission
  code/helpers/            supplied data prep, trainer, evaluation, generation,
                           plotting, submission tooling, optional W&B
  configs/laptop.json      2 layers, width 128, context 128 (development)
  configs/final.json       8 layers, width 512, context 512 (both reported runs)
  tests/test_student.py    three student tests (+ bug demonstration in notes/)
  tests/test_interfaces.py supplied smoke tests
  train.sbatch             Slurm job for one GPU training run
  prepare.sbatch           Slurm CPU job for split + tokenizer + encoding
setup.md                   local and GSC1 environment setup
gpu-check.sbatch           five-minute GPU smoke job
data/                      datasets and token files        (not in Git)
runs/                      training outputs                 (not in Git)
submissions/               packaged models and receipt      (not in Git)
report/                    report and figures
notes/                     test explanations, bug demo, defence notes
PLAN.md, CODE_MAP.md, CHECKLIST.md, RESUME.md   project notes
```

## Environment

> Local development used **Python 3.14.3** because Windows Application Control blocked
> uv-managed 3.12 on this machine; the cluster provides 3.12 and the suite is to be
> re-verified there. Torch 2.9.0+cu129 runs on the laptop's NVIDIA T500 (4 GB).

```sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat
uv pip install torch --torch-backend=auto    # cluster: --torch-backend=cu128
uv pip install -r assignment1/code/requirements.txt
```

## Reproduce

```sh
# 1. data: split whole documents, train one 4096-token BPE tokenizer, encode
python assignment1/code/tools.py split-data --input data/raw/training-data.txt \
  --out data/a1/splits --val-fraction 0.01 --seed 0
python assignment1/code/tools.py train-tokenizer --input data/a1/splits/train.txt \
  --out data/a1/tokenizer --vocab-size 4096
python assignment1/code/tools.py prepare --input data/a1/splits/train.txt \
  --val-input data/a1/splits/val.txt --out data/a1/final \
  --tokenizer data/a1/tokenizer/tokenizer.json

# 2. tests
python -m pytest assignment1/tests -q

# 3. laptop check, then the two reported runs (on one GPU via Slurm)
python assignment1/code/train.py --config assignment1/configs/laptop.json \
  --data data/a1/laptop --out runs/laptop-baseline --device cpu
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline
sbatch assignment1/train.sbatch data/a1/final runs/a1-variant --activation gelu

# 4. compare, package and upload
python assignment1/code/tools.py plot-runs runs/a1-baseline runs/a1-variant \
  --out report/comparison.png
python assignment1/code/tools.py submission package --baseline runs/a1-baseline \
  --variant runs/a1-variant --tests assignment1/tests/test_student.py \
  --out submissions/a1
```

## Models

| Model | Change | Best val bits per byte |
|---|---|---|
| baseline | SiLU gate (course default) | _pending_ |
| variant | GELU gate (`--activation gelu`) | _pending_ |

## Report and tests

- `report/` — report (at most two pages) and figures.
- `notes/` — test explanations, the bug demonstration (failure and passing output).
