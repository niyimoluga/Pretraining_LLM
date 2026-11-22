# Assignment 1 — detailed status report

**Project folder:** `C:\Users\LENOVO\Downloads\llm-a1-work` (the VS Code workspace)
**Last updated:** 2026-11-20
**Overall:** environment + data + tokenizer **done**; model/training code **not yet
implemented**; no training runs yet; GSC1 login **blocked** on credentials.

---

## 1. Quick answers to the common questions

| Question | Answer |
|---|---|
| Is the data prepared? | **Yes.** But there is **no `data/processed/`** — this course uses `data/a1/`. |
| Where are `train.bin` / `val.bin`? | `data/a1/laptop/` (small) and `data/a1/final/` (full). See §4. |
| Is the tokenizer trained? | **Yes.** `data/a1/tokenizer/tokenizer.json`, vocab 4096, `eot_id = 0`. |
| Was `laptop.json` tested? | **No.** Training cannot run yet — the 5 TODOs are still stubs. |
| Did `batch_size` need lowering for 4 GB VRAM? | Not yet tested. Estimated safe: laptop config is ~1M params, batch 8 × ctx 128; a dummy batch on the T500 used only **36.5 MB** (`see.py`). |
| Have any training runs happened? | **No.** `runs/` is empty. |
| Are the tests written? | **No.** `tests/test_student.py` is still the empty template. |
| Is anything on Hugging Face? | **No.** |
| Is the GitHub repo pushed? | **No.** Local git only (8 commits), no remote. |
| Can we log in to GSC1? | **Yes** — resolved: username is `23moluga`, passwordless via SSH key. See `notes/01-gsc1-ssh.md`. |

---

## 2. Everything done so far

### 2.1 Environment (done 2026-11-14)
- `uv` **0.13.8** at `C:\Users\LENOVO\.local\bin`.
- `.venv` in the repo root, **Python 3.14.3** (system interpreter `C:\pythonman\python.exe`).
  - *Deviation:* Windows Application Control blocks uv's downloaded Python 3.12
    (`os error 4551`) and console scripts like `hf.exe`. Workaround: venv from the
    system Python; HF CLI via `python -m huggingface_hub.cli.hf`.
- **torch 2.9.0+cu129** — `torch.cuda.is_available() == True` on the NVIDIA **T500 (4 GB)**.
- tokenizers 0.22.2 · huggingface_hub 1.33.0 · numpy · pytest 9.2.0 · matplotlib.
- Supplied test suite currently: 3 fail (`NotImplementedError`, expected), 1 passes.
- Gradient smoke test passes (`4.0`).

### 2.2 Data (done 2026-11-14)
- Downloaded pinned dataset revision `9624da983649aa6c90f4dbb260b2b9c87de5350c`
  → `data/raw/training-data.txt` (1.6 GB, **541,041 documents**).
- Split whole documents, `--val-fraction 0.01 --seed 0` → **535,601 train / 5,440 val**
  (`disjoint: true`, laptop subsets nested). Sum = 541,041 = `input_documents`.
- Trained one BPE tokenizer, vocab **4096**, on a 33.5 MB sample (11,074 docs).
- Encoded **laptop** and **final** token files (see §4).
- Report-ready numbers: **4.448 bytes/token**; random-model bpb floor **≈ 2.70**;
  500,006,912-token budget = **1.09 passes** over the training split.

### 2.3 Project scaffolding & git
- Imported the assignment starter into the repo (code, configs, tests, sbatch, docs).
- `.gitignore` excludes `.venv/`, `data/`, `runs/`, `submissions/`, `*.pt`, `logins.txt`, `session/`.
- **8 commits**, local only, no remote. See §6.

### 2.4 Security fix
- `logins.txt` was accidentally committed; **purged from all history**, moved out of the
  repo to `Downloads\logins.txt.keep`, and gitignored. Password should still be rotated.

### 2.5 GSC1 (resolved 2026-11-20)
- Username was the problem: GSC1 wants the UNIGRAZonline username **`23moluga`**.
- `Host gsc1` in `~/.ssh/config` uses `User 23moluga`; ed25519 key installed →
  **passwordless login works** (`whoami` = `23moluga`, node `IT010128`).
- Slurm account **`idea_stud`** confirmed present; `/cl_tmp` has 7.8 TB free.

### 2.12 GSC1 prepared for the final runs (2026-11-22)
- Access resolved: username **`23moluga`**, passwordless ed25519 key in `~/.ssh/config`.
- **Layout** (per the cluster guide): `$HOME` only for code/settings; everything heavy on
  `/cl_tmp/$USER`; `/usertmp` unused. Project root:
  `/cl_tmp/23moluga/llm-course/llm-assignment1` (`assignment1/`, `data/a1/final`, `logs/`, `runs/`, `.venv/`).
- Created `/cl_tmp/$USER/llm-course/{cache/{uv,huggingface,wandb},llm-assignment1}`.
- Installed uv 0.13.9; built **Python 3.12.16** venv with torch **2.11.0+cu128**, tokenizers
  0.22.2, huggingface_hub 1.33.0, wandb 0.31.2.
- Copied code (254 KB) and `data/a1/final` (**1.8 GB**, ~12.5 min).
- **Cluster test run: `pytest` → 8 passed on Python 3.12** — parity with the local 3.14.3.
- **Weights & Biases:** logged in as `niyimoluga85`; key in `~/.bashrc` and `~/.netrc` so Slurm
  jobs authenticate; `wandb.Api()` verified from the login node.
- Submitted the supplied `gpu-check.sbatch`; it is **PENDING (Resources)** because all 8
  RTX 2080 Ti on IT010146 are allocated (`AllocTRES ... gres/gpu=8`).
- Notes: `notes/gsc1-setup.md`. Open decisions before launching the two runs: time limit
  (6 h may be too short → benchmark first), parallel vs sequential, and the HF token.

### 2.11 Laptop training runs on the T500 (2026-11-22)
- Ran the laptop config on the GPU: baseline (SiLU) and variant (`--activation gelu`),
  500 steps / 512k tokens each, fp32.
- **VRAM:** peak **213 MiB** of 4096 MiB; no `batch_size` reduction needed. bfloat16 not
  native on Turing, so the trainer used float32 as intended.
- **Loss falls monotonically:** baseline val bpb 3.1655 → **2.6090**, variant → **2.5889**;
  val loss 8.3405 → 6.8742 / 6.8211. Wall 12.2 s / 13.4 s.
- Step-0 val loss identical to 4 dp (8.3405 vs 8.3404) ⇒ identical initial weights, so the
  only difference is the activation.
- Plotted with the official `plot-runs` tool → `report/laptop-comparison.png` (4 panels).
  Generation from `best.pt` runs. Full write-up: `notes/laptop-run.md`.

### 2.10 TODO 5 implemented + tests T3 + fixed-batch check (2026-11-20)
- `train_step`: train mode → zero_grad → forward in `precision_context` → finite-loss
  check → backward → `clip_grad_norm_(..., 1.0)` returning the **pre-clip** norm →
  finite check → `optimizer.step()` → `(float(loss), float(grad_norm))`.
- Tests **T3**: clipping at norm 1 returns the pre-clip norm and applies `-lr` unit
  step; non-clipping regime proves gradients are cleared; `-100` loss matches manual CE.
- Required **fixed-batch check**: loss **3.49 → 0.186** over 100 updates
  (`notes/overfit-check.txt`).
- **Bug found by the supplied test:** `GPT.forward` used `targets.view(-1)`, which fails on
  the non-contiguous slice from `make_targets`. Fixed to `.reshape(-1)`. Full suite now
  **8 passed**. Details in `notes/train-step-trace.md`.

### 2.9 TODO 4 implemented + test T2 (2026-11-20)
- `make_targets` returns `windows[:, :-1]` and `windows[:, 1:]` as slice views, so
  dtype/device/order are preserved and position i predicts window[i+1].
- Test **T2** (`test_make_targets_predicts_next_token`) checks the exact expected
  tensors plus the shift relation. Supplied suite: **5 passed, 1 failed** (only
  `train_step` remains).

### 2.8 TODO 3 implemented (2026-11-20)
- `GPT.forward` complete: input validation, embedding → blocks → final RMSNorm, tied
  unembedding (`x @ embedding.weight.T`), and float32 cross-entropy with
  `ignore_index=-100`; returns `(logits, loss_or_None)`.
- Verified: logits shape, manual-CE match, tied projection exactly, `-100` masking,
  **padding invariance** (the evaluator's protocol), gradient flow to the tied embedding,
  and rejection of 1-D / `T>context` / `T=0` inputs. Details in `notes/gpt-forward-trace.md`.
- Supplied suite now: **3 passed, 2 failed** (the two failures are TODOs 4–5).

### 2.7 TODO 2 implemented (2026-11-20)
- `Block.forward` complete: pre-norm attention + gated FFN (SwiGLU/SwiGELU) with both
  residual connections. Activation selected from `Config.activation`, so `--activation
  gelu` now works for the variant.
- Verified: residual identity, exact equation match for both activations, SiLU≠GELU,
  identical parameter count. Model sizes confirmed: laptop **950,912**, final
  **27,795,968** params. Details in `notes/block-trace.md`.

### 2.6 TODO 1 implemented (2026-11-20)
- `Attention.forward` complete in `assignment1/code/model.py` (causal MHA + RoPE).
- Verified against a naive reference loop (max diff 6e-08); causality, softmax and
  masking properties checked. Test **T1** (`test_attention_is_causal`) passes, and the
  injected `tril→triu` bug is demonstrated to fail it. Details in
  `notes/attention-trace.md` and `notes/TESTS.md`.

### 2.6 Notes / planning docs
- `PLAN.md` — full phase plan, decisions, GSC blocker, commit plan.
- `CODE_MAP.md` — what every file in `assignment1/code` is for.
- `CHECKLIST.md` — tick-box commands with status.
- `RESUME.md` — how to resume this pi session.
- `notes/00-phase0-1-setup.md` — environment + data log with the corrections.
- `notes/01-gsc1-ssh.md` — SSH diagnosis and next steps.

---

## 3. What is NOT done (the actual work ahead)

| # | Task | Where | Status |
|---|---|---|---|
| 1 | `Attention.forward` (TODO 1) | `assignment1/code/model.py` | ✅ done + test T1 |
| 2 | `Block.forward` (TODO 2, SiLU+GELU) | `assignment1/code/model.py` | ✅ done (verified) |
| 3 | `GPT.forward` (TODO 3) | `assignment1/code/model.py` | ✅ done (verified) |
| 4 | `make_targets` (TODO 4) | `assignment1/code/train.py` | ✅ done + test T2 |
| 5 | `train_step` (TODO 5) | `assignment1/code/train.py` | ✅ done + tests T3 |
| 6 | Three student tests + bug demo | `assignment1/tests/test_student.py`, `notes/` | ✅ T1–T3 + bug demo |
| 7 | Laptop baseline run | `runs/laptop-baseline` | ✅ done (T500, 213 MiB peak) |
| 8 | Laptop GELU variant run | `runs/laptop-variant` | ✅ done (213 MiB peak) |
| 8 | Final baseline + variant runs (GPU) | GSC1 | ❌ blocked |
| 9 | HF upload + receipt | — | ❌ not started |
| 10 | Report (≤2 pages) | `report/` | ❌ not started |
| 11 | GitHub push | `niyimoluga/llm-assignment1` | ❌ no remote |

> Note: `see.py` in the project root is a user-created VRAM probe (untracked). It ran
> successfully on the T500 and is a useful pre-flight check for the laptop run.

---

## 4. Output locations (full map)

### Data (`data/`, gitignored)
```
data/raw/training-data.txt             1.6 GB   pinned dataset download
data/raw/source.json                   148 KB   dataset provenance
data/a1/splits/train.txt               1.5 GB   535,601 documents
data/a1/splits/val.txt                  16 MB   5,440 documents
data/a1/splits/laptop-train.txt        2.1 MB   719 documents
data/a1/splits/laptop-val.txt           65 KB   21 documents
data/a1/splits/split.json              1.7 KB   split manifest (counts, seed, checksums)
data/a1/tokenizer/tokenizer.json       141 KB   ← THE tokenizer (vocab 4096, eot 0)
data/a1/tokenizer/tokenizer-meta.json  980 B    training provenance
data/a1/laptop/train.bin               2.4 MB   600,109 tokens   (uint32)
data/a1/laptop/val.bin                  69 KB   17,227 tokens
data/a1/laptop/meta.json               2.7 KB   counts + checksums
data/a1/laptop/tokenizer.json          141 KB   copy used by the run
data/a1/final/train.bin                1.8 GB   456,968,855 tokens
data/a1/final/val.bin                   18 MB   4,507,188 tokens
data/a1/final/meta.json                2.7 KB   counts + checksums
data/a1/final/tokenizer.json           141 KB   copy used by the run
```

### Code (tracked in git)
```
assignment1/code/model.py        ← edit (TODOs 1-3)
assignment1/code/train.py        ← edit (TODOs 4-5)
assignment1/code/components.py   supplied RMSNorm + rotate
assignment1/code/tools.py        CLI dispatcher
assignment1/code/helpers/*.py    supplied data/trainer/eval/gen/plot/submission
assignment1/configs/laptop.json  2 layers, width 128, ctx 128, 500 steps, batch 8
assignment1/configs/final.json   8 layers, width 512, ctx 512, 30,518 steps, batch 32
assignment1/tests/test_interfaces.py   supplied smoke tests
assignment1/tests/test_student.py      ← edit (your 3 tests)
assignment1/train.sbatch / prepare.sbatch / gpu-check.sbatch   Slurm jobs
```

### Outputs (created as we go)
```
runs/laptop-baseline/     metrics.jsonl, best.pt, last.pt, run.json, source/
runs/a1-baseline/         same, from the GPU baseline run
runs/a1-variant/          same, from the GPU GELU run
report/comparison.png     loss / val-loss / bpb / grad-norm plots
submissions/a1/           packaged models (baseline/ + variant/)
submissions/a1-receipt.json   HF repo link + revision
```

---

## 5. Useful facts for the report

- Split: val fraction **0.01**, seed **0**, 535,601 train / 5,440 val docs, disjoint.
- Tokenizer: 4096 vocab, `eot_id = 0`, shared by every run.
- Bytes per token: **4.448** → uniform-model bpb floor **≈ 2.70**.
- Training budget: 30,518 steps × 32 × 512 = **500,006,912 tokens** = 1.09 passes.
- bpb formula (from the supplied code): `nats/byte ÷ ln(2)`.
- Ablation: **SiLU (baseline) vs GELU (`--activation gelu`)** — same params & compute.
- Expected: very small bpb difference; overfitting is not a confounder at ~1 pass.

---

## 6. Git history (local, not pushed)

```
9969693 docs: record GSC1 SSH diagnosis ...
283f825 chore: remove credentials from the repo and ignore logins.txt
e304b8a docs: soften 3.12 parity claim ...
4b3ba00 docs: correct bpb floor (≈2.70), document count (541,041) ...
a9bc842 docs: correct checklist status ...
1225b2d docs: mark setup and data phases complete ...
1c88de1 docs: record local environment, data split, tokenizer ...
acfdf8d chore: import assignment 1 starter ...
```

---

## 7. Immediate next steps

1. ~~Implement TODO 1 (`Attention.forward`) → commit with the causality test.~~ ✅
2. TODOs 2–5 with tests, then run the supplied suite + the fixed-batch overfit check.
3. **Run the laptop config** and confirm no OOM on the T500 (record batch size used).
4. Write the three student tests and the bug demonstration.
5. Resolve GSC1 login (password reset / teaching team), then run the two GPU jobs.
6. Package + upload to Hugging Face, write the report, push to GitHub, email.
