# OOD Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate both trained surrogate models on all 133 valid OOD cases, save reusable per-case prediction HDF5 files, and isolate all OOD metrics under `evaluation_workspace/results/ood`.

**Architecture:** Add a focused OOD dataset-audit module and a standalone `test_ood.py` orchestration entry. Extend the existing evaluation context with an explicit data-source constructor, while leaving the frozen in-domain manifest loader unchanged. Reuse the existing prediction store, inference loop, hierarchical metric analysis, CSV/JSON writers, and comparison plot.

**Tech Stack:** Python 3, PyTorch, PyTorch Geometric, NumPy, h5py, CSV/JSON, unittest.

## Global Constraints

- Evaluate the 133 valid OOD HDF5 cases and record the 17 terminal failures.
- Use MeshGraphNet and Transolver checkpoints from `n=100, seed=42`.
- Never fit normalization statistics on OOD data; use checkpoint statistics only.
- Save outputs under `evaluation_workspace/results/ood/n0100_seed42`.
- Reuse a prediction HDF5 only when model, case ID, checkpoint SHA256, and time-step count match.
- Preserve `ood_generalization_workspace/run_pipeline.py` and its user-set `DRY_RUN=False`.

---

### Task 1: OOD Case Inventory and Audit Export

**Files:**
- Create: `evaluation_workspace/ood_evaluation.py`
- Create: `evaluation_workspace/tests/test_ood_evaluation.py`

**Interfaces:**
- Produces: `OodCaseInventory` dataclass with `parameter_case_ids`, `valid_case_ids`, `failed_case_ids`, `data_root`, `parameters_json`, and `audit_csv`.
- Produces: `build_ood_inventory(workspace_root: Path) -> OodCaseInventory`.
- Produces: `write_ood_case_audit(inventory, output_path: Path) -> None`.

- [x] **Step 1: Write failing inventory tests**

Create temporary parameter JSON, audit CSV, valid HDF5 fixtures, and failed-case JSON. Test sorted valid IDs, preserved audit metadata, valid/failed disjointness, full parameter coverage, required HDF5 datasets, and rejection of an unaccounted missing case.

- [x] **Step 2: Run tests and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_ood_evaluation -v`

Expected: import failure because `evaluation_workspace.ood_evaluation` does not exist.

- [x] **Step 3: Implement strict inventory construction**

Validate each participating HDF5 contains `time_steps`, `mesh/coordinates`, `mesh/connectivity`, `fields/p`, and `fields/T`; require at least two time frames; require `p/T` frame and node dimensions to match; reject duplicate case IDs and valid/failed overlap.

- [x] **Step 4: Implement atomic filtered audit export**

Read `parameter_audit.csv`, preserve all columns, select only valid IDs in inventory order, and atomically write `ood_cases.csv`. Raise when an effective case lacks an audit row.

- [x] **Step 5: Run focused tests and verify GREEN**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_ood_evaluation -v`

Expected: all tests PASS.

---

### Task 2: Explicit Evaluation Context for OOD Data

**Files:**
- Modify: `evaluation_workspace/common.py`
- Modify: `evaluation_workspace/tests/test_ood_evaluation.py`

**Interfaces:**
- Produces: `load_evaluation_context_from_cases(models, train_size, seed, device, data_root, parameters_json, case_ids, source_name) -> EvaluationContext`.
- Existing `load_evaluation_context(...)` remains the in-domain manifest entry and delegates model/dataset construction without changing snapshot verification.

- [x] **Step 1: Write a failing context-construction test**

Use a minimal valid OOD dataset and patch only checkpoint discovery/model loading boundaries needed to assert that the context dataset contains the explicit OOD IDs, `manifest["test"]` equals those IDs, and its data/parameter paths point to OOD files.

- [x] **Step 2: Run the focused test and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_ood_evaluation -v`

Expected: failure because the explicit constructor is missing.

- [x] **Step 3: Refactor common context construction minimally**

Share model-name validation, checkpoint existence/hash calculation, `FpcDataset(case_ids=...)`, graph transform, and device selection. Keep `load_evaluation_context()` manifest loading and snapshot verification exactly in its current path.

- [x] **Step 4: Run OOD and existing context-adjacent tests**

Run:

```text
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_ood_evaluation evaluation_workspace.tests.test_prediction_store evaluation_workspace.tests.test_test_pipeline -v
```

Expected: all tests PASS.

---

### Task 3: Standalone OOD Test Entry and Reusable Outputs

**Files:**
- Create: `evaluation_workspace/test_ood.py`
- Modify: `evaluation_workspace/tests/test_ood_evaluation.py`
- Modify: `evaluation_workspace/README.md`
- Generate: `evaluation_workspace/results/ood/n0100_seed42/**` (ignored runtime artifacts)

**Interfaces:**
- Produces: `main(models, train_size, seed, device, reuse_predictions, overwrite_predictions, relative_error_threshold_ratio, ood_workspace, output_root) -> None`.

- [x] **Step 1: Write failing orchestration tests**

Patch inference/analysis boundaries and assert the entry passes exactly the valid OOD IDs to prediction and analysis, writes `run_config.json` with `150/133/17`, exports `ood_cases.csv`, and chooses `results/ood/n0100_seed42` without touching the in-domain result root.

- [x] **Step 2: Run the focused test and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_ood_evaluation -v`

Expected: failure because `test_ood.main` is not implemented.

- [x] **Step 3: Implement `test_ood.py` with editable variables**

Expose at the bottom:

```python
MODELS = ["meshgraphnet", "transolver"]
TRAIN_SIZE = 100
SEED = 42
DEVICE = "auto"
REUSE_PREDICTIONS = True
OVERWRITE_PREDICTIONS = False
RELATIVE_ERROR_THRESHOLD_RATIO = 0.01
OOD_WORKSPACE = PROJECT_ROOT / "ood_generalization_workspace"
OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "ood"
```

Write `run_config.json` atomically before inference, including counts, failed IDs, source paths, selected case IDs, and checkpoint hashes. Then call the existing prediction, analysis, table-writing, and plotting functions.

- [x] **Step 4: Document startup and result reuse**

Add `python evaluation_workspace\test_ood.py`, the separate output tree, HDF5 reuse rules, and the 133/17 scope to `evaluation_workspace/README.md`.

- [x] **Step 5: Run the full repository test suite**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest discover -v`

Expected: all tests PASS.

- [x] **Step 6: Execute the real OOD evaluation**

Run: `D:/Aanconda3/envs/pinn/python.exe evaluation_workspace/test_ood.py`

Expected: predictions are written below `results/ood/n0100_seed42/predictions`; interruption can be resumed because completed HDF5 files are skipped on the next run.

- [x] **Step 7: Verify runtime artifacts and workspace isolation**

Check both model prediction directories contain 133 reusable HDF5 files, `summary.json` contains two models, `run_config.json` reports 150 parameter cases/133 tested/17 failed, `ood_cases.csv` contains 133 rows, `git diff --check` passes, and the user's `DRY_RUN=False` diff remains unchanged.
