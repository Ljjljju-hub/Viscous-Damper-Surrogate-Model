"""Calculate GT-relative metrics from saved one-step and rollout predictions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_workspace.prediction_store import read_prediction
from evaluation_workspace.relative_metrics import (
    RelativeMetricAccumulator,
    compute_relative_field_metrics,
    temperature_rise,
)


FIELD_NAMES = ("p", "T")
REPORT_FIELDS = ("p", "T", "delta_T")


def _validate_initial_fields(initial_fields: np.ndarray, node_count: int) -> np.ndarray:
    values = np.asarray(initial_fields, dtype=np.float64)
    if values.shape != (node_count, len(FIELD_NAMES)):
        raise ValueError(
            "initial_fields must have shape "
            f"[{node_count}, {len(FIELD_NAMES)}], got {values.shape}."
        )
    return values


def _field_arrays(
    prediction: np.ndarray,
    truth: np.ndarray,
    initial_fields: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    prediction_values = np.asarray(prediction, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if prediction_values.shape != truth_values.shape:
        raise ValueError(
            f"prediction/truth shape mismatch: "
            f"{prediction_values.shape} != {truth_values.shape}"
        )
    if truth_values.ndim != 3 or truth_values.shape[-1] != len(FIELD_NAMES):
        raise ValueError("prediction and truth must have shape [K, N, 2].")
    initial = _validate_initial_fields(initial_fields, truth_values.shape[1])
    return {
        "p": (prediction_values[..., 0], truth_values[..., 0]),
        "T": (prediction_values[..., 1], truth_values[..., 1]),
        "delta_T": (
            temperature_rise(prediction_values[..., 1], initial[:, 1]),
            temperature_rise(truth_values[..., 1], initial[:, 1]),
        ),
    }


def _case_metrics(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_ratio: float,
) -> dict[str, dict[str, float | int]]:
    return {
        name: compute_relative_field_metrics(
            prediction, truth, threshold_ratio=threshold_ratio
        )
        for name, (prediction, truth) in arrays.items()
    }


def _one_step_arrays(
    prediction_path: Path,
    initial_fields: np.ndarray,
) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    saved = read_prediction(prediction_path)
    return saved.case_id, _field_arrays(
        saved.prediction, saved.truth, initial_fields
    )


def calculate_one_step_case(
    prediction_path: Path,
    initial_fields: np.ndarray,
    *,
    threshold_ratio: float,
) -> dict[str, dict[str, float | int]]:
    _, arrays = _one_step_arrays(prediction_path, initial_fields)
    return _case_metrics(arrays, threshold_ratio)


def _rollout_arrays(
    prediction_path: Path,
    all_truth: np.ndarray,
) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
    case_id = str(saved.get("case_id", ""))
    field_names = tuple(saved.get("field_names", ()))
    if field_names != FIELD_NAMES:
        raise ValueError(
            f"rollout field_names must be {FIELD_NAMES}, got {field_names}."
        )
    truth_values = np.asarray(all_truth, dtype=np.float64)
    if truth_values.ndim != 3 or truth_values.shape[-1] != len(FIELD_NAMES):
        raise ValueError("all_truth must have shape [K+1, N, 2].")
    meshes = list(saved.get("meshes", []))
    if len(meshes) != truth_values.shape[0]:
        raise ValueError(
            "rollout meshes/truth time mismatch: "
            f"{len(meshes)} != {truth_values.shape[0]}."
        )
    prediction = np.stack(
        [mesh.x[:, 1:3].detach().cpu().numpy() for mesh in meshes[1:]]
    )
    if prediction.shape != truth_values[1:].shape:
        raise ValueError(
            "rollout prediction/truth shape mismatch: "
            f"{prediction.shape} != {truth_values[1:].shape}."
        )
    return case_id, _field_arrays(
        prediction,
        truth_values[1:],
        truth_values[0],
    )


def calculate_rollout_case(
    prediction_path: Path,
    all_truth: np.ndarray,
    *,
    threshold_ratio: float,
) -> dict[str, dict[str, float | int]]:
    _, arrays = _rollout_arrays(prediction_path, all_truth)
    return _case_metrics(arrays, threshold_ratio)


def _read_truth(data_root: Path, case_id: str) -> np.ndarray:
    path = data_root / f"{case_id}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GT HDF5: {path}")
    with h5py.File(path, "r") as handle:
        return np.stack(
            [handle[f"fields/{name}"][:] for name in FIELD_NAMES], axis=-1
        ).astype(np.float64)


def _new_accumulators() -> dict[str, RelativeMetricAccumulator]:
    return {name: RelativeMetricAccumulator(name) for name in REPORT_FIELDS}


def _update_accumulators(
    accumulators: dict[str, RelativeMetricAccumulator],
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_ratio: float,
) -> None:
    for field_name in REPORT_FIELDS:
        prediction, truth = arrays[field_name]
        threshold = float(np.max(np.abs(truth)) * threshold_ratio)
        accumulators[field_name].update(prediction, truth, threshold)


def _finalize_accumulators(
    accumulators: dict[str, RelativeMetricAccumulator],
) -> dict[str, dict[str, float | int]]:
    return {
        field_name: accumulators[field_name].finalize()
        for field_name in REPORT_FIELDS
    }


def _one_step_scope(
    *,
    model_name: str,
    case_ids: list[str],
    prediction_root: Path,
    data_root: Path,
    threshold_ratio: float,
) -> dict[str, dict[str, float | int]]:
    accumulators = _new_accumulators()
    for case_id in case_ids:
        all_truth = _read_truth(data_root, case_id)
        saved_case_id, arrays = _one_step_arrays(
            prediction_root / model_name / f"{case_id}.h5",
            all_truth[0],
        )
        if saved_case_id != case_id:
            raise ValueError(
                f"Prediction case_id mismatch: {saved_case_id} != {case_id}."
            )
        _update_accumulators(accumulators, arrays, threshold_ratio)
    return _finalize_accumulators(accumulators)


def _rollout_scope(
    *,
    model_name: str,
    case_ids: list[str],
    rollout_root: Path,
    data_root: Path,
    threshold_ratio: float,
) -> dict[str, dict[str, float | int]]:
    accumulators = _new_accumulators()
    for case_id in case_ids:
        all_truth = _read_truth(data_root, case_id)
        saved_case_id, arrays = _rollout_arrays(
            rollout_prediction_path(rollout_root, case_id),
            all_truth,
        )
        if saved_case_id != case_id:
            raise ValueError(
                f"Rollout case_id mismatch: {saved_case_id} != {case_id}."
            )
        _update_accumulators(accumulators, arrays, threshold_ratio)
    return _finalize_accumulators(accumulators)


def rollout_prediction_path(rollout_root: Path, case_id: str) -> Path:
    return Path(rollout_root) / f"{case_id}.pt"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(
    *,
    models: list[str],
    train_size: int,
    seed: int,
    rollout_case_count: int,
    threshold_ratio: float,
    output_path: Path,
) -> None:
    manifest_path = (
        PROJECT_ROOT / "training_workspace" / "dataset_split" / "split_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_case_ids = list(manifest["test"])
    data_root = Path(manifest["data_root"])
    prediction_root = (
        PROJECT_ROOT
        / "evaluation_workspace"
        / "results"
        / "test"
        / f"n{train_size:04d}_seed{seed}"
        / "predictions"
    )

    rollout_case_ids = None
    rollout_roots = {}
    for model_name in models:
        run_dir = (
            PROJECT_ROOT
            / "training_workspace"
            / "runs"
            / model_name
            / f"n{train_size:04d}"
            / f"seed_{seed}"
        )
        evaluation = json.loads(
            (run_dir / "evaluation.json").read_text(encoding="utf-8")
        )
        selected = list(evaluation["case_ids"])[:rollout_case_count]
        if rollout_case_ids is None:
            rollout_case_ids = selected
        elif selected != rollout_case_ids:
            raise ValueError("Models do not use the same rollout case IDs.")
        rollout_roots[model_name] = run_dir / "rollouts"
    if not rollout_case_ids:
        raise ValueError("No rollout cases selected.")

    result = {
        "definition": {
            "relative_rmse": "sqrt(sum((prediction-GT)^2) / sum(GT^2)) * 100%",
            "point_relative_error": "abs(prediction-GT) / abs(GT) * 100%",
            "temperature_rise": "delta_T(t,x) = T(t,x) - T_GT(0,x)",
            "near_zero_threshold": (
                f"per case and field: {threshold_ratio * 100:g}% * max(abs(GT))"
            ),
            "near_zero_threshold_ratio": threshold_ratio,
            "uses_training_normalization_statistics": False,
        },
        "scope": {
            "full_test_case_ids": test_case_ids,
            "rollout_case_ids": rollout_case_ids,
        },
        "models": {},
    }
    for model_name in models:
        print(f"CALCULATE {model_name}: one-step {len(test_case_ids)} cases")
        full_one_step = _one_step_scope(
            model_name=model_name,
            case_ids=test_case_ids,
            prediction_root=prediction_root,
            data_root=data_root,
            threshold_ratio=threshold_ratio,
        )
        print(
            f"CALCULATE {model_name}: one-step/rollout "
            f"{len(rollout_case_ids)} same cases"
        )
        same_one_step = _one_step_scope(
            model_name=model_name,
            case_ids=rollout_case_ids,
            prediction_root=prediction_root,
            data_root=data_root,
            threshold_ratio=threshold_ratio,
        )
        rollout = _rollout_scope(
            model_name=model_name,
            case_ids=rollout_case_ids,
            rollout_root=rollout_roots[model_name],
            data_root=data_root,
            threshold_ratio=threshold_ratio,
        )
        result["models"][model_name] = {
            "one_step_full_test_81_cases": full_one_step,
            "one_step_same_10_cases": same_one_step,
            "rollout_same_10_cases": rollout,
        }

    _atomic_write_json(Path(output_path), result)
    print(f"SAVED {Path(output_path).resolve()}")
    for model_name in models:
        scopes = result["models"][model_name]
        one_step = scopes["one_step_full_test_81_cases"]["delta_T"]
        rollout = scopes["rollout_same_10_cases"]["delta_T"]
        print(
            f"{model_name}: delta_T one_step={one_step['relative_rmse_percent']:.3f}% "
            f"rollout={rollout['relative_rmse_percent']:.3f}%"
        )


if __name__ == "__main__":
    # ======================== 实验选择 ========================
    MODELS = ["meshgraphnet", "transolver"]
    TRAIN_SIZE = 100
    SEED = 42
    ROLLOUT_CASE_COUNT = 10

    # 单点相对误差中，排除低于每工况最大绝对真值 1% 的近零点。
    THRESHOLD_RATIO = 0.01
    OUTPUT_PATH = (
        PROJECT_ROOT
        / "evaluation_workspace"
        / "results"
        / "test"
        / f"n{TRAIN_SIZE:04d}_seed{SEED}"
        / "relative_metrics.json"
    )

    main(
        models=MODELS,
        train_size=TRAIN_SIZE,
        seed=SEED,
        rollout_case_count=ROLLOUT_CASE_COUNT,
        threshold_ratio=THRESHOLD_RATIO,
        output_path=OUTPUT_PATH,
    )
