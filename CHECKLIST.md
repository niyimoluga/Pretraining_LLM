# A1 checklist and copy-paste commands

Project root (VS Code folder): `C:\Users\LENOVO\Downloads\llm-a1-work`

**Status:** Phase 0 (environment) and Phase 1 (data) done 2026-11-14 — see
`notes/00-phase0-1-setup.md`. Next: implement TODO 1 (`Attention.forward`).

## 0. Setup — done

Deviations forced by this machine: Windows Application Control blocks uv-installed
Python 3.12 and `hf.exe`, so the venv is Python **3.14.3** (system interpreter,
`C:\pythonman\python.exe`) with torch 2.9.0+cu129, and the HF CLI runs as
`python -m huggingface_hub.cli.hf`.

```powershell
irm https://astral.sh/uv/install.ps1 | iex          # done: uv 0.13.8
C:\pythonman\python.exe -m venv .venv               # done (3.14.3)
uv pip install --python .venv/Scripts/python.exe numpy pytest
uv pip install --python .venv/Scripts/python.exe torch --torch-backend=auto
uv pip install --python .venv/Scripts/python.exe -r assignment1/code/requirements.txt
```

## 1. Data — done

> split 535,601 / 5,440 docs (sum = 541,041 = `input_documents`; 1%, seed 0, disjoint) · tokenizer 4096 vocab, `eot_id = 0` ·
> laptop 600,109 / 17,227 tokens · final **456,968,855** / 4,507,188 tokens
> (500,006,912 budget = 1.09 passes) · 4.448 bytes per token → random-model floor ≈ **2.70 bpb** ( = ln(4096)/ln2 / 4.448; `evaluate` divides nats by bytes **and** by ln2 → bits)

```bash
python -m huggingface_hub.cli.hf download idealab-cs2/fundamentals-of-llms-training-data \
  training-data.txt source.json --repo-type dataset \
  --revision 9624da983649aa6c90f4dbb260b2b9c87de5350c --local-dir data/raw

python assignment1/code/tools.py split-data --input data/raw/training-data.txt \
  --out data/a1/splits --val-fraction 0.01 --seed 0
python assignment1/code/tools.py train-tokenizer --input data/a1/splits/train.txt \
  --out data/a1/tokenizer --vocab-size 4096
python assignment1/code/tools.py prepare \
  --input data/a1/splits/laptop-train.txt --val-input data/a1/splits/laptop-val.txt \
  --out data/a1/laptop --tokenizer data/a1/tokenizer/tokenizer.json
python assignment1/code/tools.py prepare \
  --input data/a1/splits/train.txt --val-input data/a1/splits/val.txt \
  --out data/a1/final --tokenizer data/a1/tokenizer/tokenizer.json
rm -rf data/a1/.a1-prepare-*        # Windows cleanup bug in the helper leaves staging dirs
```

## 2. Implement (5 TODOs)

- [ ] `model.py` `Attention.forward`
- [ ] `model.py` `Block.forward` (silu **and** gelu)
- [ ] `model.py` `GPT.forward` (tied logits, `ignore_index=-100`, shape validation)
- [ ] `train.py` `make_targets` (`windows[:, :-1]`, `windows[:, 1:]`)
- [ ] `train.py` `train_step` (zero_grad → fwd → backward → clip 1.0 → step; return pre-clip norm)

## 3. Verify

- [ ] `python -m pytest assignment1/tests -q` → 3 supplied tests pass
- [ ] overfit fixed batch (1 layer/w16/2 heads/ctx8, 100 steps) → loss drops
- [ ] laptop run: `python assignment1/code/train.py --config assignment1/configs/laptop.json --data data/a1/laptop --out runs/laptop-baseline --device cpu`
- [ ] generate text from `runs/laptop-baseline/best.pt`
- [ ] toy AdamW single-parameter trace saved in `notes/adamw-trace.txt`

## 4. Tests + bug demo

- [ ] T1 causal mask invariance
- [ ] T2 next-token alignment (hard-coded pair + manual CE)
- [ ] T3 clipping (pre-clip norm returned, update scaled) + `-100` ignored
- [ ] bug demo: `notes/bug-demo.patch`, `notes/bug-demo-fail.txt`, `notes/bug-demo-pass.txt`, `notes/TESTS.md`

## 5. GSC final runs (each must end at step 30518 / 500006912 tokens)

Blocked: SSH login rejected — see `PLAN.md` §7. While blocked, everything up to and
including both laptop runs can proceed locally.

```bash
# local
scp -r assignment1 USER@l1.gsc1.uni-graz.at:/cl_tmp/USER/llm-course/REPO/practicals/
scp -r data/a1/final USER@l1.gsc1.uni-graz.at:/cl_tmp/USER/llm-course/REPO/practicals/data/a1/
# GSC
cd /cl_tmp/$USER/llm-course/REPO/practicals && source .venv/bin/activate && mkdir -p logs
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline
sbatch assignment1/train.sbatch data/a1/final runs/a1-variant --activation gelu
squeue -u $USER; tail -f logs/llm-student-*.out
```

## 6. Compare + submit

- [ ] `tools.py plot-runs runs/a1-baseline runs/a1-variant --out report/comparison.png`
- [ ] `tools.py submission package --baseline runs/a1-baseline --variant runs/a1-variant --tests assignment1/tests/test_student.py --out submissions/a1`
- [ ] `PYTHONPATH=submissions/a1/variant/source python -B -m pytest submissions/a1/tests -q -p no:cacheprovider`
- [ ] `tools.py submission upload submissions/a1 --repo-id lanretto/llm-assignment1 --receipt submissions/a1-receipt.json`
- [ ] `tools.py submission check --receipt submissions/a1-receipt.json --out submissions/a1-downloaded --trust-code`

## 7. GitHub hand-in

- [ ] push to `github.com/niyimoluga/llm-assignment1` (repo not created yet; no `gh` CLI here)
- [ ] README: how to run, HF link, full revision ID
- [ ] `a1-receipt.json`, report (≤2 pages) in `report/`, tests + explanations + bug demo
- [ ] no checkpoints or data in Git (`.gitignore` already covers them)
- [ ] email both instructors, subject `final submission: assignment 1`

## 8. Report table to fill

| Model | Vocab | Params | Trained tokens | Best val bpb | Runtime | Device |
|---|---|---|---|---|---|---|
| baseline (SiLU) | 4096 | 28.0M (verify) | 500,006,912 | | | RTX 2080 Ti, fp32 |
| variant (GELU) | 4096 | 28.0M (verify) | 500,006,912 | | | RTX 2080 Ti, fp32 |
