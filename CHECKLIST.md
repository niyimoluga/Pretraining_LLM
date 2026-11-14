# A1 checklist and copy-paste commands

Working folder: `C:\Users\LENOVO\Downloads\llm-a1-work` (plan, notes, bug-demo logs).
Code lives in the cloned fork: `.../fundamentals-of-llms-course/practicals`.

## 0. Setup
```powershell
irm https://astral.sh/uv/install.ps1 | iex
git clone https://github.com/niyimoluga/fundamentals-of-llms-course
cd fundamentals-of-llms-course\practicals
uv python install 3.12
uv venv --python 3.12 .venv
.venv\Scripts\activate.bat
uv pip install numpy pytest
uv pip install torch --torch-backend=auto      # T500 has CUDA; use =cpu if it fails
uv pip install -r assignment1\code\requirements.txt
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

## 1. Data
```bash
hf download idealab-cs2/fundamentals-of-llms-training-data training-data.txt source.json \
  --repo-type dataset --revision 9624da983649aa6c90f4dbb260b2b9c87de5350c --local-dir data/raw

python assignment1/code/tools.py split-data --input data/raw/training-data.txt \
  --out data/a1/splits --val-fraction 0.01 --seed 0
python assignment1/code/tools.py train-tokenizer --input data/a1/splits/train.txt \
  --out data/a1/tokenizer --vocab-size 4096
python assignment1/code/tools.py tokenize --tokenizer data/a1/tokenizer/tokenizer.json \
  --text "A small bird landed."
python assignment1/code/tools.py prepare \
  --input data/a1/splits/laptop-train.txt --val-input data/a1/splits/laptop-val.txt \
  --out data/a1/laptop --tokenizer data/a1/tokenizer/tokenizer.json
# after code is frozen:
python assignment1/code/tools.py prepare \
  --input data/a1/splits/train.txt --val-input data/a1/splits/val.txt \
  --out data/a1/final --tokenizer data/a1/tokenizer/tokenizer.json
```

## 2. Implement (5 TODOs)
- [ ] `model.py` `Attention.forward`
- [ ] `model.py` `Block.forward` (silu **and** gelu)
- [ ] `model.py` `GPT.forward` (tied logits, `ignore_index=-100`, shape validation)
- [ ] `train.py` `make_targets` (`windows[:, :-1]`, `windows[:, 1:]`)
- [ ] `train.py` `train_step` (zero_grad → fwd → backward → clip 1.0 → step; return pre-clip norm)

## 3. Verify
- [ ] `python -m pytest assignment1/tests -q` → 3 passed
- [ ] overfit fixed batch (1 layer/w16/2 heads/ctx8, 100 steps) → loss drops
- [ ] laptop run: `python assignment1/code/train.py --config assignment1/configs/laptop.json --data data/a1/laptop --out runs/laptop-baseline --device cpu`
- [ ] generate text from `runs/laptop-baseline/best.pt`
- [ ] toy AdamW single-parameter trace saved in `notes/adamw-trace.txt`

## 4. Tests + bug demo
- [ ] T1 causal mask invariance
- [ ] T2 next-token alignment (hard-coded pair + manual CE)
- [ ] T3 clipping (pre-clip norm returned, update scaled) + `-100` ignored
- [ ] bug demo: `notes/bug-demo.patch`, `notes/bug-demo-fail.txt`, `notes/bug-demo-pass.txt`, `REPORT_TESTS.md`

## 5. GSC final runs (each must end at step 30518 / 500006912 tokens)
```bash
# local
env | grep -i proxy >/dev/null; cat ~/.ssh/config 2>/dev/null
scp -r assignment1 USER@l1.gsc1.uni-graz.at:/cl_tmp/USER/llm-course/REPO/practicals/
scp -r data/a1/final USER@l1.gsc1.uni-graz.at:/cl_tmp/USER/llm-course/REPO/practicals/data/a1/
# GSC
cd /cl_tmp/$USER/llm-course/REPO/practicals && source .venv/bin/activate && mkdir -p logs
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline
sbatch assignment1/train.sbatch data/a1/final runs/a1-variant --activation gelu
squeue -u $USER; tail -f logs/llm-student-*.out
```

## 6. Compare + submit
- [ ] `tools.py plot-runs runs/a1-baseline runs/a1-variant --out comparison.png`
- [ ] `tools.py submission package --baseline runs/a1-baseline --variant runs/a1-variant --tests assignment1/tests/test_student.py --out submissions/a1`
- [ ] `PYTHONPATH=submissions/a1/variant/source python -B -m pytest submissions/a1/tests -q -p no:cacheprovider`
- [ ] `tools.py submission upload submissions/a1 --repo-id USER/llm-assignment1 --receipt submissions/a1-receipt.json`
- [ ] `tools.py submission check --receipt submissions/a1-receipt.json --out submissions/a1-downloaded --trust-code`

## 7. GitHub hand-in
- [ ] README: how to run, HF link, full revision ID
- [ ] `a1-receipt.json`, report (≤2 pages PDF/MD), tests + explanations + bug demo, plots
- [ ] no checkpoints in Git (`.gitignore` them)
- [ ] email both instructors, subject `final submission: assignment 1`

## 8. Report table to fill
| Model | Vocab | Params | Trained tokens | Best val bpb | Runtime | Device |
|---|---|---|---|---|---|---|
| baseline (SiLU) | 4096 | 28.0M (verify) | 500,006,912 | | | RTX 2080 Ti, fp32 |
| variant (GELU) | 4096 | 28.0M (verify) | 500,006,912 | | | RTX 2080 Ti, fp32 |
