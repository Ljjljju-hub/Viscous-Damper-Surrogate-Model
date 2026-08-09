# Model Evaluation Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two independent Python entry points for reusable full-test prediction/metrics and selected-case temporal visualization of MeshGraphNet and Transolver.

**Architecture:** `test.py` performs teacher-forced one-step inference once and atomically stores one HDF5 per model/case, then computes metrics only from those stores. `visualize_timeseries.py` independently reads saved one-step predictions or runs autoregressive rollout, and exports combined PVD/VTU frames. Small shared modules own model adapters, HDF5 schema, metric accumulation, plotting, and VTK export.

**Tech Stack:** Python 3.12, PyTorch, PyTorch Geometric, h5py, NumPy, pandas-free CSV, matplotlib, PyVista, unittest.

## Global Constraints

- Human-facing entry points accept configuration through variables at the bottom of `test.py` and `visualize_timeseries.py`; no command-line input is required.
- Test cases come only from `training_workspace/dataset_split/split_manifest.json::test`.
- Test inference is teacher-forced one-step prediction; rollout inference is explicitly separate.
- All physical predictions stored on disk are absolute `p` and `T`, not normalized deltas.
- Dynamic positions and mesh velocity always come from `FpcDataset.get_mesh_at_time()`.
- Relative error uses only points where `abs(truth) >= 0.01 * case_max_abs_truth`; absolute metrics use every point.
- Global RMSE pools all squared errors and applies the square root once.
- Results under `evaluation_workspace/results/` are ignored by Git.
- Existing training code and completed checkpoints are not modified.

---

### Task 1: Pure Metric Aggregation

**Files:**
- Create: `evaluation_workspace/__init__.py`
- Create: `evaluation_workspace/metrics.py`
- Create: `evaluation_workspace/tests/__init__.py`
- Create: `evaluation_workspace/tests/test_metrics.py`

**Interfaces:**
- Produces: `relative_error_mask(truth: np.ndarray, threshold: float) -> np.ndarray`.
- Produces: `case_relative_threshold(truth: np.ndarray, ratio: float) -> float`.
- Produces: `MetricAccumulator(field: str, collect_absolute_errors: bool = False)` with `update(...)` and `finalize()`.
- Produces: `compute_array_metrics(prediction, truth, relative_threshold) -> dict`.
- Produces: `NormalizedMSEAccumulator(output_std: np.ndarray)` with `update(prediction, truth)` and `value`.
- Produces: extrema records containing model, field, case, time index, time, node index, x, y, truth, prediction, absolute error, relative error, and metric type.

- [ ] **Step 1: Write failing tests for aggregation semantics**

Create tests using two cases with unequal errors:

```python
def test_global_rmse_pools_squared_errors_before_root():
    accumulator = MetricAccumulator("p")
    accumulator.update(
        prediction=np.array([0.0, 2.0]),
        truth=np.array([0.0, 0.0]),
        relative_threshold=0.0,
    )
    accumulator.update(
        prediction=np.array([6.0]),
        truth=np.array([0.0]),
        relative_threshold=0.0,
    )
    self.assertAlmostEqual(
        accumulator.finalize()["rmse"], np.sqrt(40.0 / 3.0)
    )


def test_relative_error_excludes_values_below_case_threshold():
    result = compute_array_metrics(
        prediction=np.array([10.0, 2.0, 0.5]),
        truth=np.array([8.0, 1.0, 0.0]),
        relative_threshold=0.08,
    )
    self.assertEqual(result["relative_valid_count"], 2)
    self.assertEqual(result["relative_excluded_count"], 1)
    self.assertAlmostEqual(result["max_relative_error_percent"], 100.0)
```

Use `unittest` and `numpy.testing` rather than adding pytest as a dependency.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_metrics -v
```

Expected: import failure because `evaluation_workspace.metrics` does not exist.

- [ ] **Step 3: Implement streaming sums and extrema**

`MetricAccumulator.update` must flatten arrays, accumulate float64 `sum_squared_error`, `sum_absolute_error`, and `count`, maintain valid/excluded relative counts, and copy the metadata of strict maxima. `finalize` returns RMSE, MAE, max errors, counts, and threshold metadata without averaging precomputed RMSE values.

For exact P95/P99 where requested, retain float32 absolute-error chunks only when `collect_absolute_errors=True`, concatenate at finalize, and call `np.percentile(errors, [95, 99])`.

- [ ] **Step 4: Implement normalized MSE and test it**

Use:

```python
scaled_error = (prediction - truth) / output_std.reshape(1, -1)
self.sum_squared += np.square(scaled_error, dtype=np.float64).sum()
self.value_count += scaled_error.size
```

Add a two-field test proving the denominator is `2 * node_time_count`.

- [ ] **Step 5: Run metric tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add evaluation_workspace/metrics.py evaluation_workspace/tests
git commit -m "feat: add hierarchical evaluation metrics"
```

### Task 2: Shared Evaluation Context and Prediction Store

**Files:**
- Create: `evaluation_workspace/common.py`
- Create: `evaluation_workspace/prediction_store.py`
- Create: `evaluation_workspace/tests/test_prediction_store.py`

**Interfaces:**
- Produces: `EvaluationContext` dataclass with manifest, dataset, device, case IDs, checkpoints, models, graph transform, and checkpoint hashes.
- Produces: `load_evaluation_context(models, train_size, seed, device, manifest_path) -> EvaluationContext`.
- Produces: `attach_fields(mesh: Data, fields: torch.Tensor) -> Data`.
- Produces: `predict_next(model_name, model, graph, graph_transform) -> torch.Tensor`.
- Produces: `PredictionCase` dataclass matching the HDF5 schema, including frozen `output_mean` and `output_std`.
- Produces: `write_prediction_atomic(path, prediction_case, metadata) -> None`.
- Produces: `read_prediction(path) -> PredictionCase`.
- Produces: `prediction_is_reusable(path, expected_metadata) -> bool`.

- [ ] **Step 1: Write failing HDF5 schema tests**

Build a synthetic `PredictionCase` with two times, three nodes, one triangle, two fields, positions, velocity, and region. Assert round-trip equality, `complete=true`, and rejection when checkpoint SHA256 differs.

- [ ] **Step 2: Run store tests and verify RED**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_prediction_store -v
```

- [ ] **Step 3: Implement atomic HDF5 writing**

Write to `.<stem>.<pid>.partial.h5`, use gzip compression, flush and close, validate all required datasets and shapes, then publish with `os.replace`. Remove partial files in `finally`. Store scalar strings as HDF5 attributes and use `complete=true` only after all datasets exist.

- [ ] **Step 4: Implement checkpoint-aware model adapters**

`common.py` loads:

```python
SurrogateSimulator(**checkpoint["model_config"])
TransolverSimulator(**checkpoint["model_config"])
```

Validate `checkpoint["model_name"]`, load `model_state_dict`, call `eval()`, and retrieve `model.output_normalizer.std` as the frozen `[sigma_delta_p, sigma_delta_T]` vector.

For MeshGraphNet, call `prepare_graph(graph, graph_transform)` before `model.predict_next`. For Transolver, call `model.predict_next` directly. Both inputs use `attach_fields` to place node type and absolute current `p/T` in `graph.x`.

- [ ] **Step 5: Run store tests and compile common code**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_prediction_store -v
D:/Aanconda3/envs/pinn/python.exe -m py_compile evaluation_workspace/common.py evaluation_workspace/prediction_store.py
```

- [ ] **Step 6: Commit**

```powershell
git add evaluation_workspace/common.py evaluation_workspace/prediction_store.py evaluation_workspace/tests/test_prediction_store.py
git commit -m "feat: add reusable prediction storage"
```

### Task 3: Full Test Inference and Hierarchical CSV Analysis

**Files:**
- Create: `evaluation_workspace/test_pipeline.py`
- Create: `evaluation_workspace/plotting.py`
- Create: `evaluation_workspace/test.py`
- Create: `evaluation_workspace/tests/test_test_pipeline.py`

**Interfaces:**
- Produces: `predict_test_case(context, model_name, case_id) -> PredictionCase`.
- Produces: `materialize_test_predictions(context, output_dir, reuse, overwrite) -> list[Path]`.
- Produces: `EvaluationTables` dataclass containing summary, case, time, case-time, extrema, and percentile rows.
- Produces: `analyze_prediction_directory(prediction_root, threshold_ratio) -> EvaluationTables`.
- Produces: `write_evaluation_tables(tables, output_dir) -> None`.
- Produces: `plot_model_comparison(summary_rows, output_path) -> None`.
- Produces: variable-driven `test.py::main(...)`.

- [ ] **Step 1: Write failing hierarchy tests**

Create two synthetic model directories, two cases, and two target times. Assert row cardinality:

```text
summary:          models
case_metrics:     models * cases
time_metrics:     models * time_indices
case_time:        models * cases * time_indices
```

Assert global RMSE from `summary` equals direct pooling, and that the extrema row identifies the correct case/time/node coordinates.

- [ ] **Step 2: Run hierarchy tests and verify RED**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_test_pipeline -v
```

- [ ] **Step 3: Implement one-case teacher-forced inference**

Read original HDF5 `time_steps` and `fields/p,T`. For each input index `k`:

```python
input_mesh = dataset.get_mesh_at_time(case_id, times[k])
input_graph = attach_fields(input_mesh, truth[k])
prediction[k] = predict_next(model_name, model, input_graph, graph_transform)
target_mesh = dataset.get_mesh_at_time(case_id, times[k + 1])
```

Store target indices `1..K`, target times, target positions/velocity, static face/region, absolute target fields, and absolute predictions. Do not feed predictions back into the next step.

- [ ] **Step 4: Implement resumable prediction materialization**

Iterate manifest test case IDs in frozen order. Skip only reusable completed files. Print one concise line per completed/skipped case and a final model summary. Run models sequentially so one GPU is sufficient.

- [ ] **Step 5: Implement four-level metric tables**

Analyze saved HDF5 without loading neural models. Write atomically with `csv.DictWriter`:

```text
summary.csv
case_metrics.csv
time_metrics.csv
case_time_metrics.csv
extrema.csv
percentiles.csv
summary.json
```

The `time_metrics` accumulator groups by integer target index and records `physical_time_min`, `physical_time_mean`, and `physical_time_max`. The `case_time` row aggregates only the nodes in one frame. Include output-standard-deviation metadata used by normalized MSE.

- [ ] **Step 6: Implement comparison plot**

Generate `model_comparison.png` with separate panels for normalized MSE, physical RMSE, MAE, and P95 absolute error. Never plot pressure and temperature on the same numeric axis.

- [ ] **Step 7: Add the variable-driven entry**

At the bottom of `test.py`, expose only user-editable variables from the design. `main` resolves output to `results/test/n0100_seed42`, writes `run_config.json`, materializes predictions, rebuilds metrics from disk, and prints output paths.

- [ ] **Step 8: Run tests and compile**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_test_pipeline -v
D:/Aanconda3/envs/pinn/python.exe -m py_compile evaluation_workspace/test.py evaluation_workspace/test_pipeline.py evaluation_workspace/plotting.py
```

- [ ] **Step 9: Commit**

```powershell
git add evaluation_workspace/test.py evaluation_workspace/test_pipeline.py evaluation_workspace/plotting.py evaluation_workspace/tests/test_test_pipeline.py
git commit -m "feat: add reusable full-test evaluation"
```

### Task 4: Independent Time-Series Visualization

**Files:**
- Create: `evaluation_workspace/vtu_export.py`
- Create: `evaluation_workspace/visualization_pipeline.py`
- Create: `evaluation_workspace/visualize_timeseries.py`
- Create: `evaluation_workspace/tests/test_vtu_export.py`
- Create: `evaluation_workspace/tests/test_visualization_pipeline.py`

**Interfaces:**
- Produces: `TemporalComparison` dataclass containing times, positions, face, truth, model predictions, velocity, and region.
- Produces: `load_saved_one_step_sequence(...) -> TemporalComparison`.
- Produces: `rollout_sequence(...) -> TemporalComparison`.
- Produces: `export_comparison_pvd(sequence, output_dir, threshold_ratio) -> Path`.
- Produces: `write_step_metrics(sequence, output_path, threshold_ratio) -> None`.
- Produces: variable-driven `visualize_timeseries.py::main(...)`.

- [ ] **Step 1: Write failing VTU/PVD tests**

Use a two-frame, one-triangle sequence. After export, read each VTU with PyVista and assert dynamic points and fields. Parse `comparison.pvd` with `xml.etree.ElementTree` and assert both physical times and relative `frames/...` paths.

- [ ] **Step 2: Run visualization tests and verify RED**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_vtu_export evaluation_workspace.tests.test_visualization_pipeline -v
```

- [ ] **Step 3: Implement saved one-step sequence loading**

Read each selected model's prediction HDF5 and require identical case ID, target time indices, times, truth, face, and positions. Load the original HDF5 only for the `START_INDEX` initial truth and use `FpcDataset.get_mesh_at_time` for its dynamic position. Initial model fields equal the supplied truth and have zero error.

- [ ] **Step 4: Implement autoregressive rollout**

Maintain a separate `current_fields` tensor for every model. At transition `k`, reconstruct the current mesh, attach that model's current fields, predict absolute next fields, compare to HDF5 truth at `k+1`, and feed the prediction into that model's next transition. Reconstruct target positions at `k+1` for output.

- [ ] **Step 5: Implement combined VTU and structured PVD**

Create each triangle grid with:

```python
points_3d = np.column_stack([positions, np.zeros(len(positions))])
grid = pv.UnstructuredGrid({pv.CellType.TRIANGLE: face.T}, points_3d)
```

Attach truth, prediction, absolute error, NaN-masked relative error, valid masks, region, and two-component velocity padded to three components. Write PVD XML using `xml.etree.ElementTree`, not hand-built XML text.

- [ ] **Step 6: Implement frame metrics and error curve**

`step_metrics.csv` has one row per model/frame with RMSE, MAE, max absolute error, max valid relative error, valid/excluded counts, time index, physical time, and rollout horizon. `error_vs_time.png` uses separate p and T panels and distinguishes models by line style/color.

- [ ] **Step 7: Add the variable-driven visualization entry**

Validate `SOURCE_MODE in {"saved_one_step", "rollout"}`, case membership in test split for saved mode, start/step bounds, and required prediction files. Save under `results/visualization/<case>/start_<index>/<mode>`.

- [ ] **Step 8: Run visualization tests and compile**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest evaluation_workspace.tests.test_vtu_export evaluation_workspace.tests.test_visualization_pipeline -v
D:/Aanconda3/envs/pinn/python.exe -m py_compile evaluation_workspace/visualize_timeseries.py evaluation_workspace/visualization_pipeline.py evaluation_workspace/vtu_export.py
```

- [ ] **Step 9: Commit**

```powershell
git add evaluation_workspace/vtu_export.py evaluation_workspace/visualization_pipeline.py evaluation_workspace/visualize_timeseries.py evaluation_workspace/tests
git commit -m "feat: add temporal prediction visualization"
```

### Task 5: Documentation, Ignore Rules, and Real Checkpoint Smoke Test

**Files:**
- Create: `evaluation_workspace/README.md`
- Modify: `.gitignore`
- Create: `evaluation_workspace/results/.gitkeep`

**Interfaces:**
- Documents the two entry points, all variables, output schemas, aggregation formulas, relative-error threshold, ParaView workflow, reuse/resume behavior, and interpretation guidance.

- [ ] **Step 1: Add output ignore rules**

```gitignore
evaluation_workspace/results/*
!evaluation_workspace/results/.gitkeep
```

- [ ] **Step 2: Write the usage and analysis document**

Include exact `pinn` commands:

```powershell
conda activate pinn
python evaluation_workspace\test.py
python evaluation_workspace\visualize_timeseries.py
```

Explain that `summary.csv` compares overall generalization, `case_metrics.csv` identifies difficult geometries/loading cases, `time_metrics.csv` identifies difficult motion phases, `case_time_metrics.csv` drills into one case/frame, and `extrema.csv` locates isolated failures.

- [ ] **Step 3: Run all evaluation unit tests**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s evaluation_workspace/tests -p "test_*.py" -v
```

- [ ] **Step 4: Run one real-case, one-step smoke check**

Use `Case_0866`, both `best.pt` files, and `DEVICE="auto"`. Limit the internal smoke invocation to one transition and write under `evaluation_workspace/results/_smoke`. Verify both stored predictions have shape `[1, 725, 2]`, contain finite values, and their target positions differ from the static reference when motion is nonzero.

- [ ] **Step 5: Export and read one real VTU frame**

Run saved one-step visualization for the smoke result, read the VTU with PyVista, and assert required p/T truth, prediction, error, region, and velocity arrays exist and have 725 points.

- [ ] **Step 6: Run repository regression tests and diff checks**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s training_workspace/tests -p "test_*.py" -v
D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s 计算有限元数据 -p "test_*.py" -v
git diff --check
```

- [ ] **Step 7: Commit documentation**

```powershell
git add .gitignore evaluation_workspace/README.md evaluation_workspace/results/.gitkeep
git commit -m "docs: explain evaluation and visualization workflow"
```
