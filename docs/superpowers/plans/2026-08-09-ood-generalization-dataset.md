# OOD Generalization Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate, reproducible 150-case grouped out-of-domain COMSOL dataset workspace with parameter auditing, isolated restart state, VTU output, and automatic HDF5 conversion.

**Architecture:** A focused OOD parameter generator owns bounds, sampling, validation, JSON, CSV, and summary output. The existing COMSOL controller and worker gain optional workspace/model paths, so the OOD entry can reuse the proven process-isolated solve and conversion flow without copying it. The original calculation directory remains the default when no new path is supplied.

**Tech Stack:** Python 3.10+, NumPy, SciPy QMC, JSON/CSV, unittest, MPh/COMSOL, PyVista, h5py.

## Global Constraints

- Generate exactly 150 cases: 50 `geometry_ood`, 50 `loading_ood`, and 50 `material_ood`.
- Use IDs `Case_1001` through `Case_1150` in a separate workspace.
- Exactly one parameter is outside the training interval in each case; its outside distance is at most 10% of the training interval width.
- Enforce `sy >= b2 + 2*A + 10 mm` after rounding.
- Preserve the existing nested `geometry`, `loading`, and `material` JSON format.
- Keep the original `计算有限元数据` behavior unchanged when no workspace override is supplied.
- Use Python entry points only; do not add batch files.

---

## File Structure

- Create `ood_generalization_workspace/parameter_generator.py`: bounds, grouped LHS sampling, validation, and audit serialization.
- Create `ood_generalization_workspace/generate_parameters.py`: user-facing parameter generation entry with variables at the end.
- Create `ood_generalization_workspace/run_pipeline.py`: one Python entry for validation, dry-run, COMSOL execution, and HDF5 conversion.
- Create `ood_generalization_workspace/tests/test_parameter_generator.py`: deterministic generation and validation tests.
- Create `ood_generalization_workspace/tests/test_pipeline.py`: command construction and workspace-isolation tests without starting COMSOL.
- Create `ood_generalization_workspace/README.md`: Chinese operating instructions and output layout.
- Modify `计算有限元数据/main.py`: optional worker workspace and model path.
- Modify `计算有限元数据/run_remaining.py`: optional controller workspace and model path, passed to worker/converter.
- Modify `计算有限元数据/test_batch_automation.py`: regression coverage for default and overridden paths.
- Modify `.gitignore`: ignore OOD VTU, HDF5, logs, and state while retaining parameter/audit artifacts.

### Task 1: Deterministic grouped OOD parameter generation

**Files:**
- Create: `ood_generalization_workspace/parameter_generator.py`
- Create: `ood_generalization_workspace/tests/test_parameter_generator.py`

**Interfaces:**
- Produces: `GenerationConfig(sample_count_per_group: int = 50, seed: int = 20260809, outside_fraction: float = 0.1, first_case_number: int = 1001)`.
- Produces: `generate_samples(config: GenerationConfig) -> list[dict]`.
- Produces: `validate_samples(samples: list[dict], config: GenerationConfig) -> dict`.
- Produces: `write_dataset_artifacts(output_dir: Path, samples: list[dict], config: GenerationConfig, overwrite: bool) -> dict[str, Path]`.

- [ ] **Step 1: Write failing generator tests**

```python
class ParameterGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_and_has_expected_groups(self):
        config = GenerationConfig()
        first = generate_samples(config)
        second = generate_samples(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 150)
        self.assertEqual(Counter(x["ood"]["group"] for x in first), {
            "geometry_ood": 50,
            "loading_ood": 50,
            "material_ood": 50,
        })

    def test_each_case_has_exactly_one_outside_parameter_and_safe_geometry(self):
        samples = generate_samples(GenerationConfig())
        summary = validate_samples(samples, GenerationConfig())
        self.assertEqual(summary["valid_cases"], 150)
        for sample in samples:
            self.assertGreaterEqual(
                sample["geometry"]["sy"]
                - sample["geometry"]["b2"]
                - 2 * sample["loading"]["A"],
                10.0,
            )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s ood_generalization_workspace/tests -p "test_parameter_generator.py" -v`

Expected: import failure because `parameter_generator.py` does not exist.

- [ ] **Step 3: Implement bounds, balanced target assignment, LHS sampling, rejection, metadata, and strict validation**

The generator must classify values using strict training bounds, calculate:

```python
normalized_ood_distance = abs(value - nearest_training_boundary) / (upper - lower)
```

and reject any case with a second outside parameter or safety margin below 10 mm.

- [ ] **Step 4: Run generator tests and verify GREEN**

Run the command from Step 2.

Expected: all generator tests pass.

### Task 2: Artifact serialization and overwrite protection

**Files:**
- Modify: `ood_generalization_workspace/parameter_generator.py`
- Modify: `ood_generalization_workspace/tests/test_parameter_generator.py`
- Create: `ood_generalization_workspace/generate_parameters.py`

**Interfaces:**
- Consumes: `generate_samples`, `validate_samples`, and `write_dataset_artifacts` from Task 1.
- Produces: compatible `4_Combined_Master_Dataset.json`, `parameter_audit.csv`, and `dataset_summary.json`.

- [ ] **Step 1: Add failing serialization tests**

```python
def test_artifacts_round_trip_and_refuse_unrequested_overwrite(self):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config = GenerationConfig()
        samples = generate_samples(config)
        paths = write_dataset_artifacts(root, samples, config, overwrite=False)
        payload = json.loads(paths["parameters"].read_text(encoding="utf-8"))
        self.assertEqual(len(payload["parameters_list"]), 150)
        with self.assertRaises(FileExistsError):
            write_dataset_artifacts(root, samples, config, overwrite=False)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest ood_generalization_workspace.tests.test_parameter_generator -v`

Expected: failure because artifact writing is incomplete.

- [ ] **Step 3: Implement atomic JSON/CSV writes and generation entry**

`generate_parameters.py` must expose editable constants at the end:

```python
SAMPLE_COUNT_PER_GROUP = 50
RANDOM_SEED = 20260809
OUTSIDE_FRACTION = 0.10
FIRST_CASE_NUMBER = 1001
OVERWRITE = False
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2.

Expected: all tests pass and no files are written outside the temporary test directory.

### Task 3: Make the existing COMSOL runner workspace-aware

**Files:**
- Modify: `计算有限元数据/main.py`
- Modify: `计算有限元数据/run_remaining.py`
- Modify: `计算有限元数据/test_batch_automation.py`

**Interfaces:**
- Produces in `main.py`: `configure_workspace(workspace_root: Path | None, model_path: Path | None) -> None`.
- Produces in `run_remaining.py`: `configure_workspace(workspace_root: Path | None, model_path: Path | None) -> None`.
- Worker CLI accepts `--workspace-root` and `--model-path`.
- Controller CLI accepts `--workspace-root` and `--model-path` and forwards both to the worker.
- Converter command explicitly passes `--input-dir` and `--output-dir`.

- [ ] **Step 1: Add failing default/override path tests**

```python
def test_worker_command_forwards_isolated_workspace(self):
    configure_workspace(self.temp_root, self.model_path)
    command = build_worker_command(["Case_1001"], cores=4)
    self.assertIn("--workspace-root", command)
    self.assertIn(str(self.temp_root.resolve()), command)
    self.assertIn("--model-path", command)

def test_converter_command_uses_workspace_output_directories(self):
    configure_workspace(self.temp_root, self.model_path)
    command = build_converter_command(["Case_1001"])
    self.assertEqual(command[command.index("--input-dir") + 1], str(self.temp_root / "comsol_output"))
```

- [ ] **Step 2: Run automation tests and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s 计算有限元数据 -p "test_batch_automation.py" -v`

Expected: failure because workspace configuration and forwarded CLI flags do not exist.

- [ ] **Step 3: Implement explicit workspace configuration**

Keep script source paths separate from data workspace paths. Calling `configure_workspace(None, None)` must restore:

```text
计算有限元数据/4_Combined_Master_Dataset.json
计算有限元数据/standard_model.mph
计算有限元数据/comsol_output
计算有限元数据/comsol_hdf5
```

- [ ] **Step 4: Run automation tests and verify GREEN**

Run the command from Step 2.

Expected: existing restart tests and new workspace tests all pass without importing or starting MPh.

### Task 4: Unified OOD pipeline entry and dry-run isolation

**Files:**
- Create: `ood_generalization_workspace/run_pipeline.py`
- Create: `ood_generalization_workspace/tests/test_pipeline.py`

**Interfaces:**
- Consumes: generated artifacts and `计算有限元数据/run_remaining.py`.
- Produces: `build_controller_command(config: PipelineConfig) -> list[str]`.
- Produces: `run_pipeline(config: PipelineConfig) -> int`.

- [ ] **Step 1: Write failing command and validation tests**

```python
def test_controller_command_targets_only_ood_workspace(self):
    config = PipelineConfig(workspace_root=self.root, model_path=self.model, dry_run=True)
    command = build_controller_command(config)
    self.assertIn(str(self.root.resolve()), command)
    self.assertIn(str(self.model.resolve()), command)
    self.assertIn("--dry-run", command)

def test_pipeline_refuses_invalid_existing_parameter_file(self):
    self.parameters.write_text('{"parameters_list": []}', encoding="utf-8")
    with self.assertRaises(ValueError):
        validate_workspace_parameters(self.root, GenerationConfig())
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `D:/Aanconda3/envs/pinn/python.exe -m unittest ood_generalization_workspace.tests.test_pipeline -v`

Expected: import failure because `run_pipeline.py` does not exist.

- [ ] **Step 3: Implement the pipeline and editable constants**

The file end must contain:

```python
GENERATE_PARAMETERS = True
OVERWRITE_PARAMETERS = False
DRY_RUN = True
BATCH_SIZE = 10
CORES = 16
MAX_RETRIES = 2
PAUSE_SECONDS = 10.0
TIMEOUT_MINUTES = 0.0
SHOW_WORKER_WINDOW = True
```

The controller is launched through `subprocess.run(..., cwd=workspace_root, check=False)` using the current `pinn` Python executable.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run the command from Step 2.

Expected: all tests pass without starting COMSOL.

### Task 5: Documentation, generated parameters, and end-to-end dry-run

**Files:**
- Create: `ood_generalization_workspace/README.md`
- Create: `ood_generalization_workspace/4_Combined_Master_Dataset.json`
- Create: `ood_generalization_workspace/parameter_audit.csv`
- Create: `ood_generalization_workspace/dataset_summary.json`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: a ready-to-run OOD workspace with committed parameter provenance.

- [ ] **Step 1: Add OOD runtime ignore rules**

Ignore only:

```text
ood_generalization_workspace/comsol_output/
ood_generalization_workspace/comsol_hdf5/
ood_generalization_workspace/batch_logs/
ood_generalization_workspace/batch_state.json
ood_generalization_workspace/failed_cases.json
```

- [ ] **Step 2: Generate the fixed 150-case artifacts**

Run: `D:/Aanconda3/envs/pinn/python.exe ood_generalization_workspace/generate_parameters.py`

Expected: JSON, CSV, and summary report 150 valid cases with 50 cases per group.

- [ ] **Step 3: Write the Chinese README**

Document the exact two-step operation:

```text
1. Keep DRY_RUN=True and run run_pipeline.py to inspect all batches.
2. Set DRY_RUN=False and run the same Python file to calculate and convert.
```

Also explain restart behavior, failed-case behavior, output paths, group definitions, and how to change sample generation safely.

- [ ] **Step 4: Run all non-COMSOL tests**

Run:

```text
D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s ood_generalization_workspace/tests -p "test_*.py" -v
D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s 计算有限元数据 -p "test_batch_automation.py" -v
```

Expected: all tests pass.

- [ ] **Step 5: Execute the actual OOD dry-run**

Run: `D:/Aanconda3/envs/pinn/python.exe ood_generalization_workspace/run_pipeline.py`

Expected: parameters validate, 150 pending cases appear as 15 batches of 10, no COMSOL server starts, and no original training output is counted.

- [ ] **Step 6: Final integrity checks**

Run:

```text
git diff --check
git status --short
```

Expected: only intended source, documentation, test, and generated parameter artifacts are present; no VTU/HDF5/log files are staged.
