# Relative Early-Stopping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 0.2%-relative early stopping with patience 10 while preserving the checkpoint with the exact lowest validation loss.

**Architecture:** Put the state transition in a small pure-Python policy module so it can be tested without loading PyTorch datasets. The shared training loop consumes that policy, persists both exact-best and meaningful-reference state, and both model launchers receive the same configuration.

**Tech Stack:** Python 3.10+, dataclasses, unittest, PyTorch shared training pipeline.

## Global Constraints

- Exact `best.pt` saving remains based on any strict validation-loss decrease.
- Early-stop patience resets only when relative improvement is at least `0.002` (0.2%).
- Default patience is `10`.
- `ReduceLROnPlateau` behavior remains unchanged.
- Legacy checkpoints without a reference loss resume from their exact best loss.
- A running worker must be restarted before source changes take effect.

---

### Task 1: Relative Early-Stopping Policy

**Files:**
- Create: `meshGraphNet_self/early_stopping.py`
- Create: `training_workspace/tests/test_early_stopping.py`

**Interfaces:**
- Produces: `EarlyStoppingUpdate` dataclass.
- Produces: `update_early_stopping(valid_loss, best_valid_loss, reference_loss, epochs_without_improvement, min_relative_improvement) -> EarlyStoppingUpdate`.
- Produces: `reference_from_checkpoint(checkpoint, best_valid_loss) -> float`.

- [ ] **Step 1: Write failing policy tests**

```python
import math
import unittest

from meshGraphNet_self.early_stopping import (
    reference_from_checkpoint,
    update_early_stopping,
)


class EarlyStoppingTests(unittest.TestCase):
    def test_tiny_exact_improvement_does_not_reset_patience(self):
        result = update_early_stopping(0.999, 1.0, 1.0, 4, 0.002)
        self.assertTrue(result.exact_improvement)
        self.assertFalse(result.meaningful_improvement)
        self.assertEqual(result.epochs_without_improvement, 5)
        self.assertEqual(result.reference_loss, 1.0)
        self.assertEqual(result.best_valid_loss, 0.999)

    def test_threshold_improvement_resets_patience(self):
        result = update_early_stopping(0.998, 1.0, 1.0, 4, 0.002)
        self.assertTrue(result.meaningful_improvement)
        self.assertEqual(result.epochs_without_improvement, 0)
        self.assertEqual(result.reference_loss, 0.998)

    def test_first_validation_initializes_both_best_values(self):
        result = update_early_stopping(0.5, math.inf, math.inf, 0, 0.002)
        self.assertTrue(result.exact_improvement)
        self.assertTrue(result.meaningful_improvement)
        self.assertEqual(result.best_valid_loss, 0.5)
        self.assertEqual(result.reference_loss, 0.5)

    def test_legacy_checkpoint_uses_exact_best_as_reference(self):
        self.assertEqual(reference_from_checkpoint({}, 0.25), 0.25)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest training_workspace.tests.test_early_stopping -v
```

Expected: import failure because `meshGraphNet_self.early_stopping` does not exist.

- [ ] **Step 3: Implement the minimal pure policy**

```python
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EarlyStoppingUpdate:
    best_valid_loss: float
    reference_loss: float
    epochs_without_improvement: int
    exact_improvement: bool
    meaningful_improvement: bool
    relative_improvement: float


def update_early_stopping(
    valid_loss: float,
    best_valid_loss: float,
    reference_loss: float,
    epochs_without_improvement: int,
    min_relative_improvement: float,
) -> EarlyStoppingUpdate:
    if not 0.0 <= min_relative_improvement < 1.0:
        raise ValueError("min_relative_improvement must be in [0, 1).")
    exact = valid_loss < best_valid_loss
    new_best = min(best_valid_loss, valid_loss)
    if not math.isfinite(reference_loss):
        relative = math.inf
        meaningful = True
    elif reference_loss == 0.0:
        relative = 0.0
        meaningful = False
    else:
        relative = (reference_loss - valid_loss) / abs(reference_loss)
        meaningful = relative >= min_relative_improvement
    return EarlyStoppingUpdate(
        best_valid_loss=new_best,
        reference_loss=valid_loss if meaningful else reference_loss,
        epochs_without_improvement=(0 if meaningful else epochs_without_improvement + 1),
        exact_improvement=exact,
        meaningful_improvement=meaningful,
        relative_improvement=relative,
    )


def reference_from_checkpoint(checkpoint: dict, best_valid_loss: float) -> float:
    return float(checkpoint.get("early_stopping_reference_loss", best_valid_loss))
```

- [ ] **Step 4: Run policy tests and verify GREEN**

Run the command from Step 2. Expected: four tests pass.

- [ ] **Step 5: Commit policy and tests**

```powershell
git add meshGraphNet_self/early_stopping.py training_workspace/tests/test_early_stopping.py
git commit -m "feat: add relative early stopping policy"
```

### Task 2: Curve Replay and Shared Training Integration

**Files:**
- Modify: `training_workspace/tests/test_early_stopping.py`
- Modify: `meshGraphNet_self/training.py`

**Interfaces:**
- Consumes: `update_early_stopping` and `reference_from_checkpoint` from Task 1.
- Produces checkpoint key: `early_stopping_reference_loss: float`.
- Produces metric fields: `early_stop_wait`, `relative_improvement`, `meaningful_improvement`.

- [ ] **Step 1: Add a failing Transolver curve replay test**

Use epochs 79 through 88 from the completed run after initializing the reference at epoch 78:

```python
    def test_transolver_curve_stops_at_epoch_88_but_keeps_exact_best(self):
        values = [
            (79, 0.023280902321968954),
            (80, 0.023304390547467375),
            (81, 0.02385531343366908),
            (82, 0.023906205935697446),
            (83, 0.023265274478978124),
            (84, 0.02371963376812551),
            (85, 0.02304793785763883),
            (86, 0.023142618491128943),
            (87, 0.022995012351726664),
            (88, 0.0228934765019362),
        ]
        best = reference = 0.02290656498695242
        wait = 0
        exact_best_epoch = 78
        stop_epoch = None
        for epoch, value in values:
            update = update_early_stopping(value, best, reference, wait, 0.002)
            best, reference, wait = (
                update.best_valid_loss,
                update.reference_loss,
                update.epochs_without_improvement,
            )
            if update.exact_improvement:
                exact_best_epoch = epoch
            if wait >= 10:
                stop_epoch = epoch
                break
        self.assertEqual(stop_epoch, 88)
        self.assertEqual(exact_best_epoch, 88)
        self.assertAlmostEqual(reference, 0.02290656498695242)
```

- [ ] **Step 2: Run replay test and verify RED**

Expected: fail until the policy and expected equality behavior are correct.

- [ ] **Step 3: Integrate policy into training and checkpoint state**

In `run_training`:

```python
early_stopping_reference_loss = float("inf")

update = update_early_stopping(
    valid_loss,
    best_valid_loss,
    early_stopping_reference_loss,
    epochs_without_improvement,
    args.early_stopping_min_relative_improvement,
)
best_valid_loss = update.best_valid_loss
early_stopping_reference_loss = update.reference_loss
epochs_without_improvement = update.epochs_without_improvement
improved = update.exact_improvement
```

Add `early_stopping_reference_loss` to `checkpoint_state`, restore it with `reference_from_checkpoint`, and add the three policy diagnostics to the epoch print, metrics row, TensorBoard, and final summary.

- [ ] **Step 4: Run unit tests and compile checks**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest training_workspace.tests.test_early_stopping -v
D:/Aanconda3/envs/pinn/python.exe -m py_compile meshGraphNet_self/early_stopping.py meshGraphNet_self/training.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 5: Commit integration**

```powershell
git add meshGraphNet_self/training.py training_workspace/tests/test_early_stopping.py
git commit -m "feat: apply relative early stopping in training"
```

### Task 3: Launcher Configuration and Documentation

**Files:**
- Modify: `training_workspace/run_scale_study.py`
- Modify: `training_workspace/train.py`
- Modify: `training_workspace/README.md`
- Modify: `meshGraphNet_self/技术文档.md`
- Modify: `transolver_self/技术文档.md`

**Interfaces:**
- Produces CLI option: `--early-stopping-min-relative-improvement`.
- Produces launcher variable: `EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = 0.002`.
- Changes launcher default: `EARLY_STOPPING_PATIENCE = 10`.
- Produces helpers: `early_stopping_request(args) -> dict` and `early_stopping_command_args(args) -> list[str]`.

- [ ] **Step 1: Add failing configuration propagation assertions**

Extend the tests with a `SimpleNamespace` and assert both pure launcher helpers return the configured values:

```python
from types import SimpleNamespace

from training_workspace.run_scale_study import (
    early_stopping_command_args,
    early_stopping_request,
)

args = SimpleNamespace(
    early_stopping_patience=10,
    early_stopping_min_relative_improvement=0.002,
)
self.assertEqual(
    early_stopping_request(args),
    {
        "early_stopping_patience": 10,
        "early_stopping_min_relative_improvement": 0.002,
    },
)
self.assertEqual(
    early_stopping_command_args(args),
    [
        "--early-stopping-patience",
        "10",
        "--early-stopping-min-relative-improvement",
        "0.002",
    ],
)
```

- [ ] **Step 2: Run the targeted test and verify RED**

Expected: missing configuration field or CLI option.

- [ ] **Step 3: Wire the configuration through the launcher**

Add the field to `StudyConfig`, `training_workspace/train.py::main`, the user variable block, and `add_common_training_args`. Implement the two pure helpers exactly as tested, then use `request.update(early_stopping_request(args))` and `command.extend(early_stopping_command_args(args))` in `run_one`.

- [ ] **Step 4: Update user and technical documentation**

Document that 0.2% is used only for patience reset, exact `best.pt` still uses any strict minimum, and changing the setting requires restarting a running worker.

- [ ] **Step 5: Run complete verification**

```powershell
D:/Aanconda3/envs/pinn/python.exe -m unittest discover -s training_workspace/tests -p "test_*.py" -v
D:/Aanconda3/envs/pinn/python.exe -m py_compile training_workspace/train.py training_workspace/run_scale_study.py meshGraphNet_self/training.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and diff check reports no errors.

- [ ] **Step 6: Commit launcher and docs**

```powershell
git add training_workspace/run_scale_study.py training_workspace/train.py training_workspace/README.md meshGraphNet_self/技术文档.md transolver_self/技术文档.md
git commit -m "docs: configure relative early stopping"
```
