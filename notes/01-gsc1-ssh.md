# GSC1 SSH login — status (2026-11-16)

**Verdict: SSH plumbing works; the supplied credential does not authenticate.**
Access is still blocked until the u:account credential / provisioning is resolved.

## What was set up

- `~/.ssh/config` now has a `gsc1` / `l1.gsc1.uni-graz.at` block:
  `User olaniyi.moluga`, port 22, keepalive. `ssh -G gsc1` resolves correctly.
- Verified with an automated non-interactive test (OpenSSH `SSH_ASKPASS_REQUIRE=force`):
  the host is reachable, offers `publickey,password`, and the exact password bytes
  (`Ogun2026(($$`, 12 chars, verified with `od`, with and without trailing newline)
  are sent — and **rejected** for every plausible username.

## Confirmed facts

- Wiki + setup guide agree: GSC1 login = **UNIGRAZonline username** + **UniGrazOnline
  (u:account) password**. Email is `olaniyi.moluga@edu.uni-graz.at`, so the SSH user is
  `olaniyi.moluga`.
- Tried usernames `Olaniyi.Moluga`, `olaniyi.moluga`, `olaniyi`, `moluga` — all
  `Permission denied (publickey,password)` with the correct password bytes.
  => The password provided is not what GSC1 accepts (or the account isn't entitled yet).

## What this rules out

- Not network/VPN/DNS: the login node answers and returns a real auth denial.
- Not a typo in the automation: bytes verified exactly; also works with a try on the user's
  side using a manual `ssh gsc1` (type the password).

## Next steps (in order) — needs the user, not me

1. **Reset / confirm the u:account password** at `https://account.uni-graz.at`
   (the GSC password is the UniGrazOnline / u:account password; changing it can take a
   few minutes to propagate to the cluster). Try `ssh gsc1` again by hand.
2. Confirm the **username** from the account portal / u:account profile — it may differ
   from the email local part (e.g. has a suffix). The SSH config `User` can be edited.
3. If login still fails: the account may not be entitled for GSC1 / added to `idea_stud`.
   **Email the teaching team** (`lorenz.prattes@uni-graz.at`, `ruggero.lazzaroni@uni-graz.at`)
   with your full name, matriculation number and UNIGRAZonline username, asking them to
   confirm GSC1 access and the `idea_stud` Slurm association.
4. Once a **password** login works, add an **ed25519 key** so the 1.7 GB
   `scp` of `data/a1/final` and later `scp`/`sbatch` don't need a password:
   ```sh
   ssh-keygen -t ed25519 -C "gsc1" -f ~/.ssh/gsc1   # if not already present
   type %USERPROFILE%\.ssh\gsc1.pub | ssh gsc1 "cat >> ~/.ssh/authorized_keys"
   ```
   (Add `IdentityFile ~/.ssh/gsc1` to the `gsc1` block once created.)

## Security

- `logins.txt` was accidentally committed to the repo in commit `f495a0e`; it has been
  fully purged from history (verified: `git log --all -- logins.txt` empty, `git grep`
  finds no password) and the file moved out of the repo to
  `Downloads\logins.txt.keep`. `.gitignore` now ignores `logins.txt`.
- The password was exposed in git history and in chat. **Recommend rotating it** once a
  fresh one gives access, regardless.
- Mitigation: nothing has been pushed and the repo is local only.

## Impact on the assignment

Nothing in Phases 1–4 depends on GSC1. Code, tests, bug demo and both laptop runs
proceed locally. Only the two 6 h GPU runs and the HF upload need the cluster, and they
need the login sorted first — hence it's on the critical path for the deadline, not for
the work happening now.