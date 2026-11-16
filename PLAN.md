# Assignment 1 — build and train a small language model

**Plan of attack** · working folder: `C:\Users\LENOVO\Downloads\llm-a1-work`
Assignment source: `Downloads\fundamentals-of-llms-course-main\practicals\assignment1`
Fork: https://github.com/niyimoluga/fundamentals-of-llms-course
Deadline: **Sun 13 Dec 2026, 23:59 Graz** · defence 16–18 Dec 2026

---

## 1. Do I understand the assignment?

Yes. Short version: you get a partially-stubbed decoder-only GPT trainer. You must

1. **implement 5 functions** (`TODO 1–5`) — causal attention, transformer block,
   GPT forward + loss, target construction, one optimizer step;
2. **write 3 of your own tests**, and show for one of them that it catches a real
   injected bug (with before/after test output);
3. run **two full GPU trainings** (~500M tokens each): a **baseline** and **one
   modified model** (ablation), using the same data, tokenizer, seed, schedule;
4. compare them in **bits per byte** on validation, upload both checkpoints +
   tokenizer + logs + tests to **one public Hugging Face repo**, and hand in a
   **GitHub repo** with a ≤2-page report, the `a1-receipt.json` upload receipt and
   the HF link + full revision ID.

Everything else (BPE, data prep, loader, AdamW, LR schedule, checkpoints, eval,
generation, plotting, packaging, upload) is **supplied**. Distributed training,
gradient accumulation and tokenizer work are explicitly out of scope.

The graded trick is not "make it run": the supplied tests only check shapes and
that a step executes. The report + defence reward the things they *don't* check:
**causal masking, correct target alignment, correct clipping/zeroing**, and a
properly controlled ablation.

Deliverables (from `assignment.md`):

| # | Deliverable | Where |
|---|---|---|
| 1 | Working `model.py` + `train.py` (5 functions) | GitHub |
| 2 | 3 tests + explanations + bug demo (buggy diff, fail log, pass log) | GitHub |
| 3 | Both checkpoints, tokenizer, settings (`run.json`), logs (`metrics.jsonl`) | HF (public) |
| 4 | HF link, full revision ID, `a1-receipt.json` | GitHub |
| 5 | Report ≤2 pages (loss plots, results table, split info, ablation, agent-use note) | GitHub |
| 6 | README explaining how to run everything + email to both instructors | GitHub + mail |

---

## 2. Environment gaps found on this laptop (must fix first)

Checked right now:

| Item | State | Action |
|---|---|---|
| `uv` | **not installed** | install (PowerShell: `irm https://astral.sh/uv/install.ps1 \| iex`) |
| Python | 3.14.3 system | `uv python install 3.12`, `uv venv --python 3.12 .venv` — do **not** fight torch wheels on 3.14 |
| torch | not installed | `uv pip install torch --torch-backend=auto` (T500 → CUDA build) or `--torch-backend=cpu` |
| GPU | **NVIDIA T500, 4 GB** | laptop-config dev only; **final runs must go to GSC** (`idea_stud`, `gpu`, `IT010146`, RTX 2080 Ti) |
| Repo | local copy is a **zip, not a git clone** | `git clone https://github.com/niyimoluga/fundamentals-of-llms-course` and work in the clone |
| Training data | not downloaded (~1.5 GiB) | `hf download idealab-cs2/fundamentals-of-llms-training-data ...` (pinned revision, no login needed) |
| GSC account | unknown | need it for the two final runs + HF upload from GSC |
| HF account | unknown | `hf auth login` on laptop and GSC |

Laptop config runtime sanity: 2 layers / width 128 / ctx 128 / 500 steps on CPU is
minutes, on the T500 faster. That is enough to debug everything before spending
GPU-hours.

---

## 3. Phase plan

### Phase 0 — setup (≈1h, mostly downloads)
1. Install `uv`, create `.venv` with Python 3.12 inside the **cloned** fork's `practicals/`.
2. `uv pip install -r assignment1/code/requirements.txt` (+ torch build as above).
3. Run setup check (gradient smoke test), then `python -m pytest assignment1/tests -q`
   (expect 3 failures on `NotImplementedError`).
4. Download data into `practicals/data/raw/`, check `source.json`.
5. Verify GSC + HF logins; do the `gpu-check.sbatch` job so we know the partition works
   **before** the deadline week, not during it.

### Phase 1 — data (≈1h laptop CPU)
```sh
python assignment1/code/tools.py split-data --input data/raw/training-data.txt \
  --out data/a1/splits --val-fraction 0.01 --seed 0
python assignment1/code/tools.py train-tokenizer --input data/a1/splits/train.txt \
  --out data/a1/tokenizer --vocab-size 4096
python assignment1/code/tools.py tokenize --tokenizer data/a1/tokenizer/tokenizer.json \
  --text "A small bird landed."
python assignment1/code/tools.py prepare --input data/a1/splits/laptop-train.txt \
  --val-input data/a1/splits/laptop-val.txt --out data/a1/laptop \
  --tokenizer data/a1/tokenizer/tokenizer.json
```
Then, after code is final, encode the full splits into `data/a1/final` (uint32,
~400M train tokens ≈ 1.6 GB — check free disk on GSC `cl_tmp`).

**Documented decisions for the report:** `val_fraction = 0.01`, `seed = 0`
(≈5,000 validation docs). Rationale to state in two sentences: 1% keeps the
token budget for training while still giving ~5k documents / ~3M tokens, which is
more than enough to resolve bits-per-byte differences of a few thousandths between
baseline and variant; a larger val split would only slow each of the 31 evaluations.

Also verify no overlap: `split.json` document counts + the helper's `disjoint: true`
manifest flag. Keep the tokenizer **fixed** for all runs.

### Phase 2 — implement the 5 TODOs (main technical work)

Work one component at a time, each with a shape trace. Design notes:

**TODO 1 `Attention.forward`** (`x [B,T,C]`, `cos/sin` buffer `[context, D/2]`
— note `rotate` takes even/odd pairs, so `cos[:T]` is `[T, D/2]` and must be
broadcast over the head dim):
```text
q,k,v = self.qkv(x).chunk(3, -1)            # 3 × [B,T,C]
view as [B,H,T,D], apply rotate with cos[:T], sin[:T]
scores = q @ k.transpose(-2,-1) / D**0.5    # [B,H,T,T]
mask  = ones(T,T).tril().bool()             # i may only see j <= i
scores = scores.masked_fill(~mask, -inf)
p = scores.float().softmax(-1).to(v.dtype)  # float32 softmax, cast back
y = (p @ v).transpose(1,2).reshape(B,T,C)
return self.out(y)
```
Attack angles: `qkv` must be chunked in the right order (q,k,v — not interleaved),
mask direction (lower-triangular, not upper), softmax over the **last** dim, scale by
`sqrt(D)` where `D = C/H` (not `sqrt(C)`), transpose back after merging heads.
Under `bfloat16` (not used for the final fp32 runs, but the laptop may) do the
softmax in float32 — the assignment requires it.

**TODO 2 `Block.forward`** — exactly:
```text
x = x + attn(attn_norm(x))
h = ffn_norm(x)
x = x + down(act(gate(h)) * up(h))
```
with `act = F.silu` or `F.gelu` selected from `self.activation` (needed for the
easy variant) and `h` used by **both** gate and up (no second norm, no second residual).

**TODO 3 `GPT.forward`**:
1. validate `ids.ndim == 2 and 1 <= T <= context` (raise `ValueError`);
2. `x = embedding(ids)` → blocks (each with `self.cos`, `self.sin`) → `self.norm(x)`;
3. `logits = x @ self.embedding.weight.T` (**tied** weights) → `[B,T,V]`;
4. if `targets is not None`: `F.cross_entropy(logits.float().view(-1,V), targets.view(-1), ignore_index=-100)`;
5. return `(logits, loss_or_None)`; return `(logits, None)` when no targets.

**TODO 4 `make_targets(windows)`**: `return windows[:, :-1], windows[:, 1:]` —
`[B, T+1] → ([B,T], [B,T])`, same dtype/device, no `.contiguous()`/reordering tricks.
(One-liner, but the "predict the *next* token" direction is the single most common bug
and test #2 below pins it down.)

**TODO 5 `train_step`**:
```text
model.train(); optimizer.zero_grad(set_to_none=True)
with precision_context:
    _, loss = model(x, y)
if not torch.isfinite(loss): raise ...
loss.backward()
total = clip_grad_norm_(model.parameters(), 1.0)   # norm BEFORE clipping
if not torch.isfinite(total): raise ...
optimizer.step()
return float(loss), float(total)
```
Learner-relevant details: clip **after** backward and **before** step; return the
pre-clip norm (that is what `metrics.jsonl` plots and what "norms > 1.0 mean clipping
was applied" refers to); don't set the LR (caller does); don't call `.item()` inside
the graph in a way that breaks the loss; `set_to_none=True` is the cheap correct clear.

### Phase 3 — verification ladder (this is where the marks are)

1. `python -m pytest assignment1/tests -q` — supplied tests green.
2. **Overfit one fixed batch** (required by the assignment): 1 layer, width 16,
   2 heads, context 8, two fixed 9-token sequences, 100 identical updates → loss must
   drop clearly (log the curve in `notes/`). Also assert `grad_norm` shrinks/behaves.
3. Hand-check numerics with a tiny example (put in `notes/` for the defence):
   - shape trace table for `[2,8,16]` config;
   - causality probe: logits[:, :t] unchanged when token t+1..T-1 are changed;
   - `-100` targets leave loss and gradients untouched;
   - one AdamW step on a 1-parameter toy model: print `w`, `grad`, `w_after`
     (needed for defence item 3: "one parameter's gradient and value before/after").
4. `python assignment1/code/train.py --config assignment1/configs/laptop.json \
   --data data/a1/laptop --out runs/laptop-baseline --device cpu` → loss falls,
   `metrics.jsonl`, `best.pt`, `run.json`, `source/` all present.
5. Generate text from the laptop checkpoint to confirm the model isn't garbage.
6. Repeat laptop run with the **variant** switch on (`--activation gelu`), sanity-check.

### Phase 4 — the 3 student tests (`tests/test_student.py`)

Pick tests that check *behaviour the supplied tests ignore*, each with a named bug
it catches and a named bug it misses (required in the report/notes):

| Test | Checks | Catches | Misses |
|---|---|---|---|
| **T1 causality** | logits at positions `<= i` are bit-identical when future tokens change; and a *bi-directional* attention model would fail it | removed/inverted mask, `tril` vs `triu` off-by-one | scaling, head-merge, softmax-dim errors |
| **T2 next-token alignment** | `make_targets` output equals a hard-coded expected pair, and `GPT.forward(ids, targets)` loss equals a manually computed shifted cross-entropy; plus a model trained on one fixed batch must score `P(token_{t+1})` higher than `P(token_t)`. | predicting the current instead of the next token (no shift, or double shift); swapped input/target | masking, clipping |
| **T3 clipping + ignored targets** | with a loss scaled so the norm is ≫ 1: returned `grad_norm > 1.0`, and the parameter update equals `lr * (1/norm) * grad` direction (compare to a hand-applied `lr`-scaled step on a copy); second part: appending `-100` targets doesn't change the loss | returning the post-clip norm, forgetting `clip_grad_norm_`, `ignore_index` dropped, stale gradients not cleared | RoPE, block wiring |

**Bug demonstration for T1 (the deliverable "show it catches a bug"):**
- add `git diff`-style patch: `scores = scores.masked_fill(~mask, -inf)` → disabled
  (or `triu`), run `pytest tests/test_student.py -k causal -q` → **fails**, save output
  to `notes/bug-demo-fail.txt`;
- revert, rerun → **passes**, save to `notes/bug-demo-pass.txt`;
- store the patch as `notes/bug-demo.patch` and paste both outputs + short narrative
  in `REPORT_TESTS.md` (test explanations may live outside the 2-page limit).

### Phase 5 — the modification (ablation)

Recommended: **Easy — SiLU → GELU** in the gated FFN (`--activation gelu`), because
`Config` already accepts `gelu`, the block needs the branch anyway, and it is a
genuinely interesting modern-vs-classic comparison.

*Expected effect to write down before running:* GELU is a smooth, non-monotone-below-zero
approximation of the ReLU-family gate while SiLU is already smooth; in a SwiGLU-style
gated layer the gate's shape barely matters, so I expect **≈no change** in val bpb
(differences within noise, ≤0.5%), with identical parameter count and identical
compute. That is a legitimate, honest ablation result — and it makes for a strong
"we controlled everything" story. Keep SiLU as the baseline.

Wording checkpoint for the report: the training budget (500M tokens =
**1.09 passes** over the 457M-token training split) means each token is seen about
twice at most — so **overfitting is not a confounder**: train and validation loss
should stay close regardless of the gate. The expectation that *baseline vs variant*
stay close is attributed to the gate being a small architectural change with
identical parameter count and compute, **not** to the data repetition. Keep those
two reasons separate in the ablation paragraph.

Fallbacks if GELU is judged too trivial (decide with the user):
- **Medium:** untie the output projection (`nn.Linear(width, vocab, bias=False)`),
  +2.1M params (+7.6%) with a documented init matched to the embedding std; expectation:
  small improvement early, slightly better val bpb for a fixed token budget.
- **Harder:** grouped-query attention (e.g. 8 query heads / 2 KV heads), −params in
  `qkv`, expectation: similar quality, less attention compute — risky and needs care
  with the `rotate`/reshape code.

Rules to respect: same data, tokenizer, seed, steps, batch size, LR schedule,
eval interval, precision (fp32), GPU; only the one feature changes. **Keep unchanged
initial weights identical** — achieved by identical seed + construction order, and I
will verify by comparing a hash/summary of shared parameters between the two `run.json`s
/ checkpoints and record it in `notes/`.

Run both on GSC via the supplied `train.sbatch` (6 h limit, RTX 2080 Ti, fp32):
```sh
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline
sbatch assignment1/train.sbatch data/a1/final runs/a1-variant --activation gelu
```
Watch the 6 h wall clock vs. 30,518 steps; if it is tight, note it and request a
longer `--time` in the sbatch (record the change in the report, as instructed).

### Phase 6 — comparison, submission, report

1. `python assignment1/code/tools.py plot-runs runs/a1-baseline runs/a1-variant --out comparison.png`
   → training loss, val loss, val bpb, pre-clip grad norm.
2. Best val bpb from `metrics.jsonl` for each run; confirm last line is
   `step: 30518, tokens: 500006912` for both.
3. `... tools.py submission package --baseline ... --variant ... --tests assignment1/tests/test_student.py --out submissions/a1`
   then run the packaged tests against the variant source, inspect `submissions/a1/`.
4. `... tools.py submission upload submissions/a1 --repo-id USER/llm-assignment1 --receipt submissions/a1-receipt.json`
   → then `submission check --receipt ... --trust-code` to verify the uploaded revision.
5. Copy receipt + plots locally; write the README (how to run, HF link, revision ID,
   commands used) and the ≤2-page report:

```text
1  Setup: data split (val 1%, seed 0, doc counts), tokenizer 4096, hardware
2  Implementation notes: attention/block/forward/targets/step in 6-8 lines
3  Results table: vocab | params | trained tokens | best val bpb | runtime | device
4  Figures: train+val loss (2 panels), val bpb
5  Ablation: expected effect → measured → conclusion (1 short paragraph)
6  Tests: one line each (+ pointers to REPORT_TESTS.md), agent-use + sources note
```

6. Email both instructors with subject `final submission: assignment 1`.

---

## 4. Timeline (suggested)

| When | What |
|---|---|
| Day 1 | Phase 0 + 1 (env, data, tokenizer, laptop/final token files) |
| Day 2–3 | Phase 2 + 3 (5 TODOs, overfit batch, laptop run) |
| Day 4 | Phase 4 (3 tests + bug demo) |
| Day 5 | Phase 5 laptop variant check, copy to GSC, launch **baseline** run |
| Day 6 | Launch **variant** run, plot + analyse |
| Day 7 | Package/upload, README, report, email — leave ≥3 buffer days before 13 Dec |

## 5. Risks / gotchas

- **`eval_every`/`Eval` cost**: final validation is scored 31 times; a huge val split
  would dominate runtime → 1% is the sane choice, and I'll report the resulting
  document counts from `split.json`.
- **Data bugs are silent**: a double shift or non-causal mask trains fine and gives a
  suspiciously good loss. T1/T2 exist exactly for this; also compare train vs val loss
  (a leaking model shows val loss ≈ train loss and unrealistically low bpb).
- **Pre-clip grad norm > 1 means clipping fired** — plot it; a wall of clipping at the
  start is normal with warmup + 0.1 weight decay, permanent clipping is not.
- **Checkpoint format**: `best.pt` stores `tokenizer` + `eot_id` + `model_config`, so
  never move/retrain the tokenizer after the final runs.
- **Directory collisions**: the trainer refuses to reuse an `--out` dir; use fresh names.
- **GSC disk**: `data/a1/final` ≈ 1.6 GB train + small val; `runs/` holds 2 checkpoints
  × 2 runs (~110 MB each) and `source/`. Fine on `cl_tmp`, still worth checking quota.
- **Python 3.14** would break torch wheels — the venv must be 3.12.
- **Defence questions** are already listed in `assignment.md §7`; `notes/` will hold a
  one-page answer sheet with real numbers from our own run (toy AdamW step, shapes,
  gradient before/after, clipping example).

---

## 6. Decisions (confirmed 2026-11-14)

| Question | Decision |
|---|---|
| Variant | **GELU** (`--activation gelu`), SiLU baseline |
| Split | `--val-fraction 0.01 --seed 0` (course default, comparable) |
| GitHub | **fresh repo `niyimoluga/llm-assignment1`** (course code copied in, clean history) |
| Commits | **incremental**: one commit per TODO + its test + a short note |
| HF account | `lanretto` → target repo `lanretto/llm-assignment1` |
| GSC access | **blocked** — `Permission denied (publickey,password)` for `Olaniyi.Moluga@l1.gsc1.uni-graz.at` (see §7) |

Because the variant only flips one activation, the diffusion of work is small: the
only code difference is the `self.activation` branch in `Block.forward` (needed by
the baseline anyway) and the `--activation gelu` flag. The ablation is therefore a
clean single-variable comparison with **identical parameter count and compute**.

Commit plan (repo `niyimoluga/llm-assignment1`):

```text
chore: import course starter (assignment1 code, configs, tests, sbatch, setup docs)
feat(model): causal multi-head attention with RoPE        + tests/test_attention.py
test: causal mask invariance (T1)                          [bug demo patch + logs]
feat(model): transformer block with SwiGLU/GELU gate
feat(model): GPT forward, tied logits and CE loss          + test T-ignore/-100
feat(train): next-token targets (make_targets)             + test T2 alignment
feat(train): one optimizer update with clipping            + test T3 clipping
feat(model): --activation gelu ablation switch             (variant)
results: laptop baseline + laptop variant runs (plots, metrics)
results: final baseline + final variant runs, comparison.png
submission: HF package/receipt, README, report
```

## 7. GSC1 login blocker (critical path)

Symptom:

```text
ssh Olaniyi.Moluga@l1.gsc1.uni-graz.at
Permission denied (publickey,password).
```

Diagnosis: the login node is **reachable** and it **offers** password auth (the
`(publickey,password)` list comes from the server), so this is not a network/VPN
or hostname problem — the username/password pair was rejected.

Most likely causes, in order:

1. **Username is not `Firstname.Lastname`.** The GSC1 login name is the
   **UNIGRAZonline username**, which is often lowercase and/or shortened (e.g.
   `moluga`, `o.moluga`, or a code shown in your u:account profile). `ssh` sends the
   name exactly as typed; check `ssh -v` output line `Authentications that can
   continue:` / `password (password)`. Get the exact name from
   UNIGRAZonline → *Profile / u:account* (it is also the local part of your
   `@edu.uni-graz.at` address).
2. **u:account password not the one you think.** Cluster login uses the central
   u:account password, not the UNIGRAZonline SSO variant. Reset/verify it at
   `https://account.uni-graz.at`. Password changes can take a few minutes to propagate.
3. **u:account not activated / not entitled for GSC1.** New accounts sometimes need
   activation; GSC access must be requested/allowed for the course. The setup guide
   says to contact the teaching team if login fails — they can check the `idea_stud`
   Slurm association, which you can then verify after login with
   `sacctmgr -n show assoc user="$USER" format=Account,Partition,QOS`.
4. **Key-only access.** The node may accept passwords in general but not for your
   account once an SSH key is registered. Then you must install the key through the
   HPC portal or `ssh-copy-id` from a machine that can still log in.

Action list (do these in order, keep the exact error text):

```powershell
ssh -v Olaniyi.Moluga@l1.gsc1.uni-graz.at   # note which username is sent + methods offered
ssh <exact-u:account-username>@l1.gsc1.uni-graz.at
```

- [ ] Confirm the exact username from UNIGRAZonline (try lowercase).
- [ ] Reset/verify the u:account password at `https://account.uni-graz.at`.
- [ ] Try once more; if it still fails, **email the teaching team**
      (`lorenz.prattes@uni-graz.at`, `ruggero.lazzaroni@uni-graz.at`) with the
      command, exact output, your full name, matriculation number and UNIGRAZonline
      username, asking whether GSC1 access / `idea_stud` is provisioned.
- [ ] Once a password login works, immediately add a key so later `scp` of
      `data/a1/final` (~1.7 GB) does not need interactive passwords:
      `ssh-keygen -t ed25519 -C "gsc1"` then `type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh USER@l1.gsc1.uni-graz.at "cat >> ~/.ssh/authorized_keys"`.

**Mitigation while blocked:** everything in Phases 1–4 and the laptop runs happen
locally, so the deadline work (code, tests, bug demo, laptop baseline/variant) is
not gated on GSC. Only Phases 5–6 need the cluster, and the two 6 h GPU jobs need a
few days of buffer.

## 8. Local verification results (checked 2026-11-14)

| Check | Result |
|---|---|
| `ssh` to l1.gsc1 | reachable, auth rejected |
| `git --version` | 2.56.0 (Windows) |
| fork `niyimoluga/fundamentals-of-llms-course` | reachable, `main` = `39ca4a4` |
| free disk on C: | 208 GB (enough: data ~3 GB, runs ~0.5 GB) |
| `uv` / `torch` / Python | uv missing, torch missing, system Python 3.14 → use 3.12 venv |
| GPU | T500 4 GB (laptop config only) |
