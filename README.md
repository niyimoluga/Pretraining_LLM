# Assignment 1: build and train a small language model

Course: *Fundamentals of LLMs* (University of Graz).
Student: Olaniyi Moluga · Hugging Face: [lanretto](https://huggingface.co/lanretto)

This repository contains the implementation, tests, logs, report and commands for
Assignment 1: a small decoder-only GPT (28M parameters, 4096-token BPE vocabulary)
pre-trained on ~500M tokens twice — a **SiLU baseline** and a **GELU variant** —
and compared in validation bits per byte.

> Status: work in progress. Sections below are filled in as the assignment phases
> complete. Models will be published at `lanretto/llm-assignment1` on Hugging Face
> (link + full revision ID go here once uploaded).

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
