# Temperature Rise Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible temperature-rise relative metrics and explain P50/P95 model performance in the concise project report.

**Architecture:** Put array-level metric math in a small importable module and keep filesystem/model-result orchestration in a separate Python entry. Reuse saved one-step HDF5 predictions and rollout PT files, load only the GT initial temperature from the original case HDF5, and write one canonical JSON consumed by the report.

**Tech Stack:** Python 3, NumPy, h5py, PyTorch, unittest, Markdown.

## Global Constraints

- Define `delta_T(t, x) = T(t, x) - T_GT(0, x)` per case.
- Use prediction and GT only; never use training normalization statistics for these relative metrics.
- Use `1% * max(abs(delta_T_GT))` as the per-case near-zero threshold for point-relative temperature-rise errors.
- Reuse existing predictions; do not run model inference.
- Preserve the user's `ood_generalization_workspace/run_pipeline.py` configuration.

---

### Task 1: Array-Level Relative Metric Core

**Files:**
- Create: `evaluation_workspace/relative_metrics.py`
- Create: `evaluation_workspace/tests/test_relative_metrics.py`

**Interfaces:**
- Produces: `compute_relative_field_metrics(prediction, truth, *, threshold_ratio) -> dict`
- Produces: `temperature_rise(temperature, initial_temperature) -> np.ndarray`
- Produces: `RelativeMetricAccumulator.update(prediction, truth, threshold)`, `finalize() -> dict`

- [x] **Step 1: Write failing tests for relative RMSE, percentiles, threshold counts, and temperature rise**

Use hand-computable arrays to assert global relative RMSE, P50/P95/P99/max, valid/excluded counts, and broadcasting of `[N]` initial temperature across `[K, N]` values. Also assert shape mismatches and all-zero denominators raise clear `ValueError` exceptions.

- [x] **Step 2: Run the focused tests and verify they fail**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_relative_metrics -v`

Expected: FAIL because `evaluation_workspace.relative_metrics` does not exist.

- [x] **Step 3: Implement the minimal reusable metric core**

The accumulator stores `sum(error^2)`, `sum(truth^2)`, count, valid/excluded point counts, and point-relative chunks. `finalize()` returns:

```python
{
    "count": int,
    "absolute_rmse": float,
    "gt_rms": float,
    "relative_rmse_percent": float,
    "point_relative_valid_count": int,
    "point_relative_excluded_count": int,
    "point_relative_p50_percent": float,
    "point_relative_p95_percent": float,
    "point_relative_p99_percent": float,
    "point_relative_max_percent": float,
}
```

- [x] **Step 4: Run the focused tests and verify they pass**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_relative_metrics -v`

Expected: all tests PASS.

---

### Task 2: Saved-Prediction Calculation Entry

**Files:**
- Create: `evaluation_workspace/calculate_relative_metrics.py`
- Modify: `evaluation_workspace/tests/test_relative_metrics.py`
- Generate: `evaluation_workspace/results/test/n0100_seed42/relative_metrics.json` (ignored result artifact)

**Interfaces:**
- Consumes: `read_prediction(path) -> PredictionCase`
- Consumes: rollout dictionaries containing `meshes`, where each mesh has `x[:, 1] = p` and `x[:, 2] = T`
- Produces: `calculate_one_step_case(path, initial_fields, threshold_ratio) -> dict`
- Produces: `calculate_rollout_case(path, truth, threshold_ratio) -> dict`

- [x] **Step 1: Add failing tests for one-step and rollout extraction**

Build temporary HDF5/PT fixtures with one initial frame and two target frames. Assert both paths use the same GT initial temperature, produce identical absolute temperature errors before and after conversion to `delta_T`, and reject mismatched case/time dimensions.

- [x] **Step 2: Run tests and verify the new cases fail**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_relative_metrics -v`

Expected: FAIL because the entry helpers are not implemented.

- [x] **Step 3: Implement orchestration and explicit editable configuration**

The script locates:

```text
evaluation_workspace/results/test/n0100_seed42/predictions/<model>/Case_*.h5
training_workspace/runs/<model>/n0100/seed_42/rollouts/Case_*.pt
training_workspace/dataset_split/split_manifest.json
```

At the bottom expose `MODELS`, `TRAIN_SIZE`, `SEED`, `ROLLOUT_CASE_COUNT`, `THRESHOLD_RATIO`, and `OUTPUT_PATH` as editable variables, consistent with the project's no-command-line training/test entries.

The JSON must include `p`, absolute `T`, and `delta_T` for full 81-case one-step and same-10-case one-step/rollout scopes, plus metric definitions and case IDs. Write atomically through a temporary file and `os.replace`.

- [x] **Step 4: Run focused and existing evaluation tests**

Run:

```text
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_relative_metrics evaluation_workspace.tests.test_metrics evaluation_workspace.tests.test_prediction_store -v
```

Expected: all tests PASS.

- [x] **Step 5: Execute the calculation using saved predictions**

Run: `D:/Aanconda3/envs/pinn/python.exe evaluation_workspace/calculate_relative_metrics.py`

Expected: no model inference messages; JSON contains finite `delta_T` metrics for both models and all three scopes.

---

### Task 3: Concise Report Update and Verification

**Files:**
- Modify: `项目总结与面试复习.md`
- Modify: `docs/superpowers/specs/2026-08-09-interview-review-document-design.md`

**Interfaces:**
- Consumes: `evaluation_workspace/results/test/n0100_seed42/relative_metrics.json`

- [x] **Step 1: Replace misleading absolute-temperature percentages with temperature-rise results**

Keep temperature absolute RMSE in K, add `delta_T` relative RMSE, and state that absolute-temperature percentages use roughly 297 K as the denominator and therefore understate temperature-rise difficulty.

- [x] **Step 2: Explain P50/P95 and interpret the measured model behavior**

Add plain-language definitions: P50 is the typical median point; P95 bounds 95% of valid node-time errors. Explain single-step pressure core accuracy, pressure tail errors, temperature-rise accuracy, and rollout drift separately using the computed values.

- [x] **Step 3: Verify every displayed value against JSON**

Use a PowerShell JSON check to assert report strings match the rounded source values and that `uses_training_normalization_statistics` is false.

- [x] **Step 4: Run final verification**

Run:

```text
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_relative_metrics evaluation_workspace.tests.test_metrics evaluation_workspace.tests.test_prediction_store -v
git diff --check
git status --short
```

Expected: tests and formatting pass; the user's `DRY_RUN=False` change remains present and unstaged.
