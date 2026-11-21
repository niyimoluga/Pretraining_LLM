# GSC1 SSH access — RESOLVED (2026-11-20)

**Status: working, passwordless, via SSH key.**

## What was wrong

The username was wrong. GSC1 logins use the **UNIGRAZonline username**, which here is
**`23moluga`** (not `olaniyi.moluga` / `Olaniyi.Moluga`). The password in `logins.txt`
was fine all along.

## What is configured now

`~/.ssh/config`:

```text
Host l1.gsc1.uni-graz.at gsc1
    HostName l1.gsc1.uni-graz.at
    User 23moluga
    IdentityFile C:\Users\LENOVO\.ssh\gsc1
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Ed25519 key `~/.ssh/gsc1` generated and installed in `~/.ssh/authorized_keys` on GSC1.

## Verified facts

| Check | Result |
|---|---|
| `ssh gsc1` with key only (`BatchMode`) | works, no password |
| `whoami` | `23moluga` |
| login node | `IT010128.uni-graz.at` |
| home directory | `/usr/people/BZEDVZ/23moluga` |
| Slurm association | `idea_stud` present (`normal` QOS) |
| `/cl_tmp` free space | 423 TB total, **7.8 TB free** (99% used — plenty for our ~2 GB) |

## Usage for the full runs (when we get there)

```sh
# from the practicals-equivalent folder on GSC1
mkdir -p /cl_tmp/$USER/llm-course
# copy code + final data (passwordless now)
scp -r assignment1 23moluga@l1.gsc1.uni-graz.at:/cl_tmp/23moluga/llm-course/llm-assignment1/practicals/
scp -r data/a1/final 23moluga@l1.gsc1.uni-graz.at:/cl_tmp/23moluga/llm-course/llm-assignment1/practicals/data/a1/
# on GSC1: build the venv, then submit
sbatch assignment1/train.sbatch data/a1/final runs/a1-baseline
sbatch assignment1/train.sbatch data/a1/final runs/a1-variant --activation gelu
```

(The exact repository folder name on GSC1 is to be decided at that point; see `PLAN.md` §5.)

## Security

`logins.txt` remains out of the repo (`Downloads\logins.txt.keep`, gitignored, purged
from history). The password was exposed earlier in git history/chat, so **rotating it
is still advisable** — but it is no longer needed for day-to-day work thanks to the key.
