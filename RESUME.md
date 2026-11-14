# How to resume this pi session

**Session ID:** `01a1dbac-8dc0-73fc-9ead-28722db8122d`
**Working directory:** `C:\Users\LENOVO`
**Created:** 2026-11-14 21:08 local (20:08 UTC) — topic: Assignment 1 (small-LLM pre-training) plan

## Live copy (pi manages this; it auto-saves on every message)

```
C:\Users\LENOVO\.pi\agent\sessions\--C--Users-LENOVO--\2026-11-14T20-08-01-217Z_01a1dbac-8dc0-73fc-9ead-28722db8122d.jsonl
```

## Resume

From a terminal in `C:\Users\LENOVO` (the session's working directory):

```bat
pi -c                                  :: continue most recent session for this dir
pi -r                                  :: pick from the list (search by id / name)
pi --session 01a1dbac                  :: by partial session ID
pi --session "C:\Users\LENOVO\.pi\agent\sessions\--C--Users-LENOVO--\2026-11-14T20-08-01-217Z_01a1dbac-8dc0-73fc-9ead-28722db8122d.jsonl"
```

Inside a running pi, `/resume` opens the picker, `/session` shows the current file/ID,
`/tree` moves to an earlier point, `/fork` and `/clone` copy the branch.

## Backups in this folder

| File | What it is |
|---|---|
| `a1-planning-01a1dbac.jsonl` | byte-exact snapshot of the session (restorable) |
| `a1-planning-01a1dbac.html` | readable HTML export of the same snapshot |

Snapshot taken after the planning turn (1 file + 2 tool verifications).

## Restore from the snapshot (if the live file is lost)

Copy the backup back into the sessions directory, keeping the original filename
pattern `<UTC-timestamp>_<session-id>.jsonl`:

```bat
copy "C:\Users\LENOVO\Downloads\llm-a1-work\session\a1-planning-01a1dbac.jsonl" ^
     "C:\Users\LENOVO\.pi\agent\sessions\--C--Users-LENOVO--\2026-11-14T20-08-01-217Z_01a1dbac-8dc0-73fc-9ead-28722db8122d.jsonl"
pi -r
```

(Verified: the snapshot re-exports cleanly, so it is a valid session file.)

## Make it easy to find

- In the running TUI, type `/name assignment1-plan` to label this session; the picker
  can then filter by name.
- `/export` re-exports at any time; `/share` gives a viewer link (don't share it — it
  contains local paths and command output).

## Recommended habit

Re-snapshot after each big milestone:

```bat
copy "C:\Users\LENOVO\.pi\agent\sessions\--C--Users-LENOVO--\2026-11-14T20-08-01-217Z_01a1dbac-8dc0-73fc-9ead-28722db8122d.jsonl" ^
     "C:\Users\LENOVO\Downloads\llm-a1-work\session\a1-planning-01a1dbac.jsonl"
```

Note: sessions are **not** portable across machines by default (they are just JSONL,
so they can be copied, but the recorded working directory must exist or pi will ask
to continue/fork). Keep the plan itself in `PLAN.md` / `CHECKLIST.md` — those, not the
transcript, are what survives a machine switch.
