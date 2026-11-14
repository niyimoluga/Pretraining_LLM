# Phase 0–1 log: environment and data (local, 2026-11-14)

Machine: Windows, NVIDIA T500 (4 GB) — laptop configuration only; final runs go to GSC1.

## Environment actually built

| Item | Result |
|---|---|
| `uv` | 0.13.8 at `C:\Users\LENOVO\.local\bin` (installed with the official PowerShell script) |
| Python | **3.14.3** (`C:\pythonman\python.exe`) — see note below |
| venv | `.venv` in the repo root, created with the *system* Python |
| torch | **2.9.0+cu129**, `torch.cuda.is_available() == True` (T500), gradient smoke test `4.0` |
| tokenizers | 0.22.2 · huggingface_hub 1.33.0 · numpy, pytest 9.2.0, matplotlib |
| supplied tests | 3 failed (`NotImplementedError` for TODO 3/4/5, expected), 1 passed (`test_bpe_interfaces`) |

### Two Windows quirks worth recording

1. **`uv python install 3.12` produces binaries that Windows Application Control blocks**
   (`os error 4551`: "An Application Control policy has blocked this file"), and therefore
   `uv venv --python 3.12` creates a venv that cannot execute. Workaround used: create
   the venv from the already-installed system Python (`python -m venv .venv`) and install
   packages with `uv pip install --python .venv/Scripts/python.exe ...`.
   The venv is **Python 3.14.3**, not 3.12; torch 2.9 has cp314 wheels, so everything works.
2. **Console scripts (`hf.exe`) are also blocked** by the same policy. Workaround:
   `python -m huggingface_hub.cli.hf download ...`.

Also: the supplied `helpers/data.py` fails to remove its staging directory on Windows
(`PermissionError: WinError 32` on `documents.jsonl`). This is cosmetic — outputs are
written before cleanup — but it leaves a multi-GB `.a1-prepare-*` directory that must be
removed manually (`rm -rf data/a1/.a1-prepare-*`). Done twice, ~3.7 GB reclaimed.

## Data

Download (pinned revision `9624da983649aa6c90f4dbb260b2b9c87de5350c`): `data/raw/training-data.txt`,
1.6 GB, **529,318** `<|endoftext|>` document separators.

| Stage | Command | Result |
|---|---|---|
| split | `split-data --val-fraction 0.01 --seed 0` (45 s) | 535,601 train docs (1.59 GB) / 5,440 val docs (15.7 MB); `disjoint: true`, laptop subsets nested in their parents |
| laptop subset | supplied defaults | 719 train docs (2.1 MB) / 21 val docs (64 KB) |
| tokenizer | `train-tokenizer --vocab-size 4096` (9 min) | 11,074 docs / 33.55 MB sample, vocab **4096**, `eot_id = 0`, sha256 `118cfb9a…` |
| tokenize check | `tokenize "A small bird landed."` | `A | small | b | ird | land | ed | .` = 7 tokens, round-trip clean |
| laptop encode | `prepare … --out data/a1/laptop` | 600,109 train / 17,227 val tokens |
| final encode | `prepare … --out data/a1/final` (8 min) | **456,968,855** train / 4,507,188 val tokens; `train.bin` 1.8 GB (`<u4`) |

Notes for the report:

- mean **4.448 UTF-8 bytes per token**, so the random-model baseline for bits per byte is
  ≈ ln(4096)/ln(2) / 4.448 ≈ **1.87**; a trained 28M model should be well below 1.
- the 500,006,912-token budget is **1.09 passes** over the 456.97M training tokens, i.e. each
  token is seen about once — this explains why both runs stay close together.
- validation text is 15,733,430 bytes, scored 31 times per final run (~4.5M tokens each).
