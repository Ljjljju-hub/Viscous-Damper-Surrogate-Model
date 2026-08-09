# Interview Review Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a concise Chinese project summary and interview review sheet using only verified experiment results.

**Architecture:** Recompute all derived percentages and speedups from the saved JSON/CSV records, then write one root Markdown document that separates one-step accuracy, autoregressive rollout, and limitations. Verify every reported number against its source before completion.

**Tech Stack:** Markdown, Python standard library for metric checks, existing training/evaluation JSON and CSV artifacts.

## Global Constraints

- Create only `项目总结与面试复习.md` plus this implementation plan.
- Do not modify the user's current `DRY_RUN=False` setting.
- Label all results as `n=100, seed=42, RTX 4060` where applicable.
- Keep one-step teacher-forced and 150-step autoregressive metrics separate.
- Compute relative RMSE and point-relative errors from prediction and GT only; do not use training normalization statistics.
- Mark OOD accuracy as pending until COMSOL and model evaluation finish.

---

### Task 1: Recompute derived comparison values

**Files:**
- Read: `training_workspace/runs/*/n0100/seed_42/summary.json`
- Read: `evaluation_workspace/results/test/n0100_seed42/summary.json`
- Read: `training_workspace/runs/*/n0100/seed_42/evaluation.json`

- [x] Recompute parameter, training-time, one-step error, rollout error, and speed differences with Python.
- [x] Record the exact experimental denominator for every value.

### Task 2: Write the interview review sheet

**Files:**
- Create: `项目总结与面试复习.md`

- [x] Write the 30-second introduction and experimental scope.
- [x] Add the compact model comparison table.
- [x] Explain one-step accuracy, point-error tails, rollout stability, and speed in short conclusions.
- [x] Add four core technical decisions and concise interview Q&A.
- [x] End with limitations, next steps, and a five-line memory card.

### Task 3: Verify wording and numbers

**Files:**
- Verify: `项目总结与面试复习.md`

- [x] Search for unsupported claims and mixed metric scopes.
- [x] Re-read every number against JSON/CSV sources.
- [x] Run `git diff --check` and confirm the user's OOD runtime change remains untouched.

### Task 4: Add GT-relative accuracy metrics

**Files:**
- Read: `evaluation_workspace/results/test/n0100_seed42/predictions/<model>/*.h5`
- Read: `training_workspace/runs/<model>/n0100/seed_42/rollouts/*.pt`
- Modify: `项目总结与面试复习.md`

- [x] Compute global relative L2 RMSE as `sqrt(sum(error^2) / sum(GT^2)) * 100%`.
- [x] Compute point-relative P50/P95/P99/max with the existing per-case `1% * max(abs(GT))` threshold.
- [x] Report the valid and excluded point counts.
- [x] Compare one-step and rollout on the same 10 test cases.
- [x] Keep absolute RMSE only as an auxiliary physical-unit metric.
