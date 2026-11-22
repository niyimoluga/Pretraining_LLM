# GSC1 setup for the final runs (2026-11-22)

## Access

- Host: `l1.gsc1.uni-graz.at` (alias `gsc1` in `~/.ssh/config`)
- User: **`23moluga`** (UNIGRAZonline username)
- Auth: ed25519 key (`~/.ssh/gsc1`) — passwordless
- Slurm account: **`idea_stud`**; partitions `hpc` (CPU prep) and `gpu` (training)

## Storage layout — what lives where

The wiki/setup guide is explicit:

| Location | Purpose | Backup | Used for |
|---|---|---|---|
| `$HOME` = `/usr/people/BZEDVZ/23moluga` | code and settings only, **space limited** | yes | `.bashrc`, `.netrc`, `.ssh`, `~/.local/bin` (uv) |
| `/cl_tmp/$USER` | working files, datasets, environment, results | **no backup** | everything heavy: repo, data, `.venv`, runs |
| `/usertmp/$USER` | per-compute-node temp | no | not used for this course |

Actual free space at setup time: `/cl_tmp` 423 TB total, **11 TB free** (98 % used cluster-wide);
`/usr/people` 39 TB with 26 TB free. Our footprint is tiny by comparison (≈2 GB data + ≈3 GB venv).

```
/cl_tmp/23moluga/llm-course/
├── cache/
│   ├── uv/                      uv package cache   (UV_CACHE_DIR)
│   ├── huggingface/hub/         HF downloads      (HF_HUB_CACHE)
│   └── wandb/                   W&B local files   (WANDB_DIR)
└── llm-assignment1/             <- project root, mirrors the laptop layout
    ├── assignment1/             code, configs, tests, sbatch   (254 KB)
    ├── data/a1/final/           train.bin 1.8 GB, val.bin 18 MB, meta.json, tokenizer.json
    ├── logs/                    Slurm .out/.err
    ├── runs/                    a1-baseline/, a1-variant/
    └── .venv/                   Python 3.12 environment
```

`$HOME/.bashrc` gains a marked block exporting `PATH`, `UV_CACHE_DIR`, `HF_HUB_CACHE`,
`WANDB_DIR`, `WANDB_CACHE_DIR` and `WANDB_API_KEY` (see §Weights & Biases).

## Environment on the cluster

| Item | Value |
|---|---|
| Python | **3.12.16** (via `uv python install 3.12`) |
| torch | **2.11.0+cu128** |
| tokenizers / huggingface_hub | 0.22.2 / 1.33.0 |
| wandb | 0.31.2 |
| uv | 0.13.9 (`~/.local/bin/uv`) |
| Login node | IT010128, 24 CPUs, 64 GB RAM |
| **Tests on cluster** | `pytest assignment1/tests -q` → **8 passed** ✅ |

This also settles the Python-version question: the suite passes on **3.12** as well as the
local **3.14.3**, so the code has no version-specific behaviour.

## Weights & Biases

- Account: **`niyimoluga85`**, default entity `niyimoluga85-universty-of-technology-graz`.
- This wandb build resolves `base_url = https://api.forge.coreweave.com` (verified with
  `wandb.Settings().base_url`), and `~/.netrc` holds the key for that host.
- Login verified from the login node: `wandb.Api().viewer.username == "niyimoluga85"`.
- API key is exported in `~/.bashrc` **and** stored in `~/.netrc` (so Slurm jobs, which do not
  source `.bashrc`, still authenticate).
- Usage: add `--wandb-project <name>` to the training command; local `metrics.jsonl` is still
  written regardless, so W&B is additive and non-blocking.

> Security: the key was shared in chat, so treat it as exposed and rotate it in the W&B UI
> when convenient. It is stored only on the cluster (`~/.bashrc`, `~/.netrc`), never in the repo.

## GPU partitions (observed)

| Node | GRES | Note |
|---|---|---|
| IT010146 | `gpu:rtx2080ti:8` | the assignment's target node (11–12 GB per card) |
| IT010147 | `gpu:a100.80gb:8` | much faster, not the course default |
| IT010149 | `gpu:h200:4` | fastest, not the course default |
| IT010213 | `gpu:a100.40gb:2` | idle at setup time |

`train.sbatch` defaults to `--gres=gpu:rtx2080ti:1 --nodelist=IT010146 --time=06:00:00`.

## Run plan (pending user sign-off)

```sh
cd /cl_tmp/23moluga/llm-course/llm-assignment1
mkdir -p logs
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline --wandb-project llm-a1
sbatch assignment1/train.sbatch data/a1/final runs/a1-variant  --activation gelu --wandb-project llm-a1
```

Open questions to settle before launching:

1. **Time limit.** 6 h may be too short. Rough estimate: 6·N·T ≈ 6 × 27.8e6 × 5e8 ≈ 8.3e16 FLOP;
   at a realistically usable ~4 TFLOP/s fp32 on a 2080 Ti that is ≈5.8 h before attention
   overhead — i.e. right at the default limit. A short benchmark job will measure the real
   seconds/step, then `--time` is set with margin (e.g. `12:00:00`).
2. **Parallel vs sequential.** Two jobs each requesting one 2080 Ti may run concurrently on
   different cards (halves wall time) or queue one after the other.
3. **HF upload.** `hf auth login` needs a token (or a browser code) for user `lanretto`.