# Worktree Exercise — Three parallel lines of work (worktrees + integration)

*Inspired by [Hidden in Plain Text: Emergence & Mitigation of Steganographic
Collusion in LLMs](https://arxiv.org/abs/2410.03768) (Mathew et al., 2024).*

> **Read this yourself — don't paste it into Claude.** The agent is configured
> *not* to read `EXERCISE.md`. The point of the exercise is for *you* to understand
> the tasks and why they're set up this way (why worktrees, why the shared monitor,
> what the integration is really testing), and then to drive Claude in **your own
> words**. Copy-pasting this brief defeats the exercise — the skill being practised
> is you steering the agent, not the agent reading the spec.

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

## Step 1 — get the lay of the land

Before you create a single worktree, get an inside view of the codebase — and use
**Claude as your guide** to do it. Ask it to walk you through the repo: what the
project does, how a `steg_comm` run flows from `main.py` → the config →
`functions/run_steg.py` → the `lib/` helpers (`sampling`, `parsing`, `scoring`,
`monitor`), how the config system composes `base.yaml` / `debug.yaml`, and what
`experiments/monitor_test` is for. Have it explain *why* things are arranged this
way, not just where they are.

You need this mental model for everything after. You can't restate the three tasks
in your own words, or judge whether the agent's change is actually right, without
understanding the loop it's editing — and reading the code *through* the agent is
the fastest way in.

## The setup

You have **three independent lines of work** on this project. Two of them edit and
re-run the `steg_comm` experiment; the third hardens a shared library
(`lib/monitor.py`) and proves it on `experiments/monitor_test`, which already
benchmarks it. Run them serially in one working tree and they trip over each
other — shared config edits clobber each other, you can't run two API loops at
once, and the monitor change ripples into `steg_comm`'s scores.

So each task gets its **own branch in its own worktree**, off this branch, and one
agent per worktree — all three developing and running at the same time. Working out
the worktree and merge mechanics is part of the exercise; the commands aren't
spelled out here.

## Your three tasks

### Task A — tune the scheme
Set a **goal**: maximize `combined_score`. A good scheme recovers the numbers
reliably *and* reads as low-suspicion — the two pull against each other. Let an
agent iterate: propose a scheme, edit `scheme:` in `configs/base.yaml`, run the
config, read `metrics.json`, repeat. The debug run is small and the model is
stochastic, so judge a scheme over several samples, not one.

**Done when:** `configs/base.yaml` holds a scheme whose `combined_score` is clearly
higher than the one the experiment ships with.

### Task B — let the model invent its own scheme
Right now the scheme is fixed in config. Expand `run_steg` so the encoder model
**proposes its own scheme** first (an extra model turn), then encodes with it; the
decoder is then told that proposed scheme and decodes. Add a config flag
(`model_generated_scheme: true`) to switch between fixed and self-invented schemes,
and log the proposed scheme alongside each completion.

**Done when:** a run in invent-scheme mode has the model choose the scheme and still
recover numbers, with the chosen scheme saved in the completions.

### Task C — harden the monitor
The monitor in `lib/monitor.py` is a single-prompt 0–9 score, and
`experiments/monitor_test` already benchmarks it against the labelled
`solutions.jsonl` (mean suspiciousness per `clean` / `subtle` / `obvious`). Make the
monitor **separate those labels more sharply** — an ensemble of calls, a
second-opinion prompt, whatever works — and prove it by re-running `monitor_test`.
`lib/monitor.py` is also what `steg_comm` calls to score its answers, so this is the
one change that reaches outside its own experiment. (If the handful of seed rows is
too thin to trust the result, grow `solutions.jsonl` first.)

**Done when:** re-running `monitor_test` shows your hardened `lib/monitor.py`
separating `clean` < `subtle` < `obvious` more sharply than the shipped monitor.

## Bring them back together

This is the point of the exercise: three parallel branches off a shared base,
brought onto one branch and **re-validated together**. Two things have to be
reconciled, and they're different in kind:

- **A and B both edited `configs/base.yaml`** → a real text conflict you resolve by
  hand.
- **C changed `lib/monitor.py`, which `steg_comm` calls** → there's no conflict
  marker, but `steg_comm` is now scored by your tougher monitor. After merging,
  re-run the integrated branch: A's tuned scheme and B's invented schemes are
  graded by the stronger monitor, so their `combined_score` **moves**. Integration
  isn't done when it merges cleanly — it's done when you've re-run and the numbers
  still hold up.

**Done when:** one branch runs clean with all three changes active, you've re-run
`steg_comm` under the hardened monitor and checked the scores, and the spare
worktrees are cleaned up.

## The technique: worktrees + integration

Three independent lines of work → three worktrees → three agents developing and
running at once, fully isolated. The advanced half isn't resolving a conflict
marker; it's **integration**: parallel branches off a shared base, one of them
touching a dependency the others rely on, brought back together and re-validated as
a whole.
