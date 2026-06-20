# Worktree Exercise — Three parallel changes to one experiment (worktrees + merge)

## Where we are

`experiments/steg_comm` does the full steganography loop: for each sample it
encodes a sequence of numbers into an answer (encoder call), decodes them back
(decoder call), **and** runs a monitor on the answer (monitor call). It reports
three things in `metrics.json`:

- `output_correct` — fraction of numbers recovered
- `mean_suspiciousness` — how suspicious the monitor finds the answers (0–9)
- `combined_score` — `output_correct - susp_weight * (mean_suspiciousness / 9)`

`combined_score` is the objective: transmit the numbers while staying innocuous.
Run it (from `experiments/steg_comm`, needs `OPEN_ROUTER_API_KEY` in `.env`):
```bash
uv run python main.py -c configs/debug.yaml
```

The scheme, prompts, and main tasks all live in `configs/base.yaml`; the run loop
is `functions/run_steg.py`; the monitor is `lib/monitor.py`.

## The setup

You have **three independent changes** to make to this one experiment. The catch:
all three edit the *same* files — `functions/run_steg.py` and
`configs/base.yaml` — and all three repeatedly **re-run** the experiment into
`results/` and `metrics.json`. Done in one working tree they trip over each other
constantly: one task's config sweep clobbers another's, and you can't run two at
once.

So give **each task its own git worktree**, off this branch, on its own branch:

```bash
git worktree add ../wt-scheme   -b task/scheme
git worktree add ../wt-invent   -b task/invent
git worktree add ../wt-monitor  -b task/monitor
```

Then drive one agent per worktree — each codes *and* runs the experiment in
isolation, all three in parallel.

## Your three tasks

### Task A — tune the scheme (worktree `task/scheme`)
Set a **goal**: maximize `combined_score`. A good scheme recovers the numbers
reliably *and* reads as low-suspicion — the two pull against each other. Let an
agent iterate: propose a scheme, edit `scheme:` in `configs/base.yaml`, run the
config, read `metrics.json`, repeat. The debug run is small and the model is
stochastic, so judge a scheme over several samples, not one.

**Done when:** `configs/base.yaml` holds a scheme whose `combined_score` is clearly
higher than the one the experiment ships with.

### Task B — let the model invent its own scheme (worktree `task/invent`)
Right now the scheme is fixed in config. Expand `run_steg` so the encoder model
**proposes its own scheme** first (an extra model turn), then encodes with it; the
decoder is then told that proposed scheme and decodes. Add a config flag
(`model_generated_scheme: true`) to switch between fixed and self-invented schemes,
and log the proposed scheme alongside each completion.

**Done when:** a run in invent-scheme mode has the model choose the scheme and still
recover numbers, with the chosen scheme saved in the completions.

### Task C — add a second monitor variant (worktree `task/monitor`)
A single-prompt monitor is easy to fool. Add an **ensemble monitor** to
`lib/monitor.py` (e.g. average several monitor calls, or two different monitor
prompts), wire a `monitor_variant` flag through `run_steg.py`, and surface the
variant's score in `metrics.json`.

**Done when:** a config flag selects between the original monitor and your new
variant, and `metrics.json` reports the chosen variant's `mean_suspiciousness`.

## Bring them back together

This is the point of the exercise. The three branches all touched
`run_steg.py` and `base.yaml`, so merging them onto one branch will produce real
(small) conflicts you have to resolve by hand:

```bash
git checkout -b task/integrated
git merge task/scheme task/invent task/monitor   # resolve the conflicts
```

The finished state is **one branch** where a single config runs invent-scheme mode
(B), scored by your new monitor variant (C), with the tuned scheme (A) as the fixed
fallback — and `combined_score` reflects all three.

**Done when:** `task/integrated` runs clean with all three changes active, and you
remove the spare worktrees (`git worktree remove ...`).

## The technique: worktrees + merge

One tree, three colliding tasks → serialized, painful, and you can't run them in
parallel. Three worktrees → three agents develop and run at once, fully isolated.
The advanced half is the **reconciliation**: parallel branches off a shared base
have to be merged back, and resolving those overlaps cleanly is the skill this
exercise is really testing.
