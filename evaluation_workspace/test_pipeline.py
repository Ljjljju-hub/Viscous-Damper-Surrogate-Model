from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import EvaluationContext, attach_fields, predict_next
from .metrics import (
    MetricAccumulator,
    NormalizedMSEAccumulator,
    case_relative_threshold,
    compute_array_metrics,
    relative_error_mask,
)
from .prediction_store import PredictionCase, prediction_is_reusable, read_prediction, write_prediction_atomic


FIELD_NAMES = ("p", "T")


@dataclass
class EvaluationTables:
    summary: list[dict]
    case_metrics: list[dict]
    time_metrics: list[dict]
    case_time_metrics: list[dict]
    extrema: list[dict]
    percentiles: list[dict]


def _read_case_fields(context: EvaluationContext, case_id: str):
    file_path = context.dataset._resolve_file(case_id)
    with h5py.File(file_path, "r") as handle:
        times = handle["time_steps"][:].astype(np.float64)
        truth = np.stack(
            [handle[f"fields/{name}"][:] for name in FIELD_NAMES], axis=-1
        ).astype(np.float32)
    return times, truth


@torch.no_grad()
def predict_test_case(
    context: EvaluationContext,
    model_name: str,
    case_id: str,
    max_steps: int | None = None,
) -> PredictionCase:
    loaded = context.get_model(model_name)
    times, all_truth = _read_case_fields(context, case_id)
    predictions = []
    positions = []
    velocity = []
    face = None
    region = None
    available_steps = len(times) - 1
    step_count = available_steps if max_steps is None else min(max_steps, available_steps)
    if step_count < 1:
        raise ValueError("max_steps must permit at least one prediction.")
    for input_index in range(step_count):
        input_mesh = context.dataset.get_mesh_at_time(case_id, float(times[input_index]))
        input_fields = torch.as_tensor(all_truth[input_index], dtype=torch.float32)
        graph = attach_fields(input_mesh, input_fields)
        next_fields = predict_next(
            model_name, loaded, graph, context.graph_transform
        ).detach().cpu().numpy()
        target_mesh = context.dataset.get_mesh_at_time(
            case_id, float(times[input_index + 1])
        )
        predictions.append(next_fields.astype(np.float32, copy=False))
        positions.append(target_mesh.pos.numpy())
        velocity.append(target_mesh.mesh_velocity.numpy())
        if face is None:
            face = target_mesh.face.numpy()
            region = target_mesh.mesh_region.numpy()
    return PredictionCase(
        case_id=case_id,
        model_name=model_name,
        checkpoint_path=str(loaded.checkpoint_path),
        checkpoint_sha256=loaded.checkpoint_sha256,
        time_indices=np.arange(1, step_count + 1, dtype=np.int64),
        time_steps=times[1 : step_count + 1],
        positions=np.stack(positions),
        velocity=np.stack(velocity),
        face=np.asarray(face),
        region=np.asarray(region),
        truth=all_truth[1 : step_count + 1],
        prediction=np.stack(predictions),
        output_mean=loaded.output_mean.numpy(),
        output_std=loaded.output_std.numpy(),
    )


def materialize_test_predictions(
    context: EvaluationContext,
    prediction_root: Path,
    *,
    reuse: bool,
    overwrite: bool,
    case_ids: list[str] | None = None,
    max_steps: int | None = None,
) -> list[Path]:
    prediction_root = Path(prediction_root)
    selected_cases = case_ids or list(context.manifest["test"])
    output_paths = []
    for model_name in context.model_names:
        model_dir = prediction_root / model_name
        for case_number, case_id in enumerate(selected_cases, 1):
            output_path = model_dir / f"{case_id}.h5"
            with h5py.File(context.dataset._resolve_file(case_id), "r") as source:
                expected_time_count = len(source["time_steps"]) - 1
            if max_steps is not None:
                expected_time_count = min(max_steps, expected_time_count)
            reusable = prediction_is_reusable(
                output_path,
                model_name=model_name,
                case_id=case_id,
                checkpoint_sha256=context.checkpoint_hashes[model_name],
                expected_time_count=expected_time_count,
            )
            if reusable and reuse and not overwrite:
                print(f"SKIP {model_name} {case_id} ({case_number}/{len(selected_cases)})")
            else:
                print(f"PREDICT {model_name} {case_id} ({case_number}/{len(selected_cases)})")
                prediction = predict_test_case(
                    context, model_name, case_id, max_steps=max_steps
                )
                write_prediction_atomic(output_path, prediction)
            output_paths.append(output_path)
    return output_paths


def _field_columns(field_name: str, metrics: dict) -> dict:
    return {f"{field_name}_{key}": value for key, value in metrics.items()}


def _row_metrics(
    data: PredictionCase,
    time_slice=None,
    collect=False,
    threshold_ratio: float = 0.01,
) -> dict:
    prediction = data.prediction if time_slice is None else data.prediction[time_slice]
    truth = data.truth if time_slice is None else data.truth[time_slice]
    result = {}
    for field_index, field_name in enumerate(FIELD_NAMES):
        threshold = case_relative_threshold(
            data.truth[..., field_index], threshold_ratio
        )
        field_metrics = compute_array_metrics(
            prediction[..., field_index],
            truth[..., field_index],
            threshold,
            collect_absolute_errors=collect,
        )
        field_metrics["relative_threshold"] = threshold
        result.update(_field_columns(field_name, field_metrics))
    normalized = NormalizedMSEAccumulator(data.output_std)
    normalized.update(prediction, truth)
    result["normalized_mse"] = normalized.value
    return result


def _extreme_record(
    data: PredictionCase,
    field_index: int,
    metric_type: str,
    threshold: float,
) -> dict | None:
    truth = data.truth[..., field_index]
    prediction = data.prediction[..., field_index]
    absolute = np.abs(prediction - truth)
    if metric_type == "absolute":
        values = absolute
    else:
        valid = relative_error_mask(truth, threshold)
        if not valid.any():
            return None
        values = np.full(truth.shape, -np.inf, dtype=np.float64)
        values[valid] = absolute[valid] / np.abs(truth[valid]) * 100.0
    flat_index = int(np.argmax(values))
    time_offset, node_index = np.unravel_index(flat_index, values.shape)
    return {
        "model": data.model_name,
        "field": FIELD_NAMES[field_index],
        "metric_type": metric_type,
        "case_id": data.case_id,
        "time_index": int(data.time_indices[time_offset]),
        "physical_time": float(data.time_steps[time_offset]),
        "node_index": int(node_index),
        "x": float(data.positions[time_offset, node_index, 0]),
        "y": float(data.positions[time_offset, node_index, 1]),
        "truth": float(truth[time_offset, node_index]),
        "prediction": float(prediction[time_offset, node_index]),
        "absolute_error": float(absolute[time_offset, node_index]),
        "relative_error_percent": (
            float(values[time_offset, node_index])
            if metric_type == "relative"
            else (
                float(absolute[time_offset, node_index] / abs(truth[time_offset, node_index]) * 100.0)
                if abs(truth[time_offset, node_index]) >= threshold
                and truth[time_offset, node_index] != 0
                else math.nan
            )
        ),
        "relative_threshold": threshold,
    }


def analyze_prediction_directory(
    prediction_root: Path,
    *,
    threshold_ratio: float = 0.01,
    models: list[str] | tuple[str, ...] | None = None,
    case_ids: list[str] | tuple[str, ...] | None = None,
) -> EvaluationTables:
    prediction_root = Path(prediction_root)
    summary_rows = []
    case_rows = []
    case_time_rows = []
    extrema_rows = []
    percentile_rows = []
    time_rows = []
    model_dirs = (
        [prediction_root / name for name in models]
        if models is not None
        else sorted(path for path in prediction_root.iterdir() if path.is_dir())
    )
    for model_dir in model_dirs:
        files = (
            [model_dir / f"{case_id}.h5" for case_id in case_ids]
            if case_ids is not None
            else sorted(model_dir.glob("Case_*.h5"))
        )
        missing_files = [str(path) for path in files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Missing prediction files:\n" + "\n".join(missing_files[:20])
            )
        if not files:
            continue
        global_fields = {
            name: MetricAccumulator(name, collect_absolute_errors=True)
            for name in FIELD_NAMES
        }
        time_fields: dict[int, dict[str, MetricAccumulator]] = {}
        time_normalized: dict[int, NormalizedMSEAccumulator] = {}
        time_values: dict[int, list[float]] = {}
        global_normalized = None
        global_extrema: dict[tuple[str, str], dict] = {}
        output_std = None
        for file_path in files:
            data = read_prediction(file_path)
            if output_std is None:
                output_std = data.output_std
                global_normalized = NormalizedMSEAccumulator(output_std)
            elif not np.allclose(output_std, data.output_std):
                raise ValueError(f"output_std changed within {model_dir.name} predictions.")
            thresholds = [
                case_relative_threshold(data.truth[..., index], threshold_ratio)
                for index in range(len(FIELD_NAMES))
            ]
            case_row = {"model": data.model_name, "case_id": data.case_id}
            case_row.update(
                {
                    "output_std_p": float(data.output_std[0]),
                    "output_std_T": float(data.output_std[1]),
                }
            )
            case_row.update(
                _row_metrics(
                    data, collect=True, threshold_ratio=threshold_ratio
                )
            )
            case_rows.append(case_row)
            global_normalized.update(data.prediction, data.truth)
            for field_index, field_name in enumerate(FIELD_NAMES):
                global_fields[field_name].update(
                    data.prediction[..., field_index],
                    data.truth[..., field_index],
                    thresholds[field_index],
                )
                case_metrics = compute_array_metrics(
                    data.prediction[..., field_index],
                    data.truth[..., field_index],
                    thresholds[field_index],
                    collect_absolute_errors=True,
                )
                percentile_rows.append(
                    {
                        "scope": "case",
                        "model": data.model_name,
                        "case_id": data.case_id,
                        "field": field_name,
                        "p95_absolute_error": case_metrics["p95_absolute_error"],
                        "p99_absolute_error": case_metrics["p99_absolute_error"],
                        "max_absolute_error": case_metrics["max_absolute_error"],
                    }
                )
                for metric_type in ("absolute", "relative"):
                    record = _extreme_record(
                        data, field_index, metric_type, thresholds[field_index]
                    )
                    if record is None:
                        continue
                    extrema_rows.append({"scope": "case", **record})
                    key = (field_name, metric_type)
                    compare_key = (
                        "absolute_error"
                        if metric_type == "absolute"
                        else "relative_error_percent"
                    )
                    if key not in global_extrema or record[compare_key] > global_extrema[key][compare_key]:
                        global_extrema[key] = record

            for offset, time_index in enumerate(data.time_indices):
                time_index = int(time_index)
                frame_row = {
                    "model": data.model_name,
                    "case_id": data.case_id,
                    "time_index": time_index,
                    "physical_time": float(data.time_steps[offset]),
                }
                frame_row.update(
                    _row_metrics(
                        data,
                        time_slice=offset,
                        threshold_ratio=threshold_ratio,
                    )
                )
                case_time_rows.append(frame_row)
                if time_index not in time_fields:
                    time_fields[time_index] = {
                        name: MetricAccumulator(name) for name in FIELD_NAMES
                    }
                    time_normalized[time_index] = NormalizedMSEAccumulator(data.output_std)
                    time_values[time_index] = []
                for field_index, field_name in enumerate(FIELD_NAMES):
                    time_fields[time_index][field_name].update(
                        data.prediction[offset, :, field_index],
                        data.truth[offset, :, field_index],
                        thresholds[field_index],
                    )
                time_normalized[time_index].update(
                    data.prediction[offset], data.truth[offset]
                )
                time_values[time_index].append(float(data.time_steps[offset]))

        summary = {
            "model": model_dir.name,
            "case_count": len(files),
            "output_std_p": float(output_std[0]),
            "output_std_T": float(output_std[1]),
        }
        for field_name in FIELD_NAMES:
            finalized = global_fields[field_name].finalize()
            summary.update(_field_columns(field_name, finalized))
            percentile_rows.append(
                {
                    "scope": "global",
                    "model": model_dir.name,
                    "case_id": "ALL",
                    "field": field_name,
                    "p95_absolute_error": finalized["p95_absolute_error"],
                    "p99_absolute_error": finalized["p99_absolute_error"],
                    "max_absolute_error": finalized["max_absolute_error"],
                }
            )
        summary["normalized_mse"] = global_normalized.value
        summary_rows.append(summary)
        extrema_rows.extend(
            {"scope": "global", **record} for record in global_extrema.values()
        )
        for time_index in sorted(time_fields):
            values = time_values[time_index]
            row = {
                "model": model_dir.name,
                "time_index": time_index,
                "physical_time_min": min(values),
                "physical_time_mean": sum(values) / len(values),
                "physical_time_max": max(values),
            }
            for field_name in FIELD_NAMES:
                row.update(
                    _field_columns(field_name, time_fields[time_index][field_name].finalize())
                )
            row["normalized_mse"] = time_normalized[time_index].value
            time_rows.append(row)
    return EvaluationTables(
        summary_rows,
        case_rows,
        time_rows,
        case_time_rows,
        extrema_rows,
        percentile_rows,
    )


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_evaluation_tables(tables: EvaluationTables, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    mappings = {
        "summary.csv": tables.summary,
        "case_metrics.csv": tables.case_metrics,
        "time_metrics.csv": tables.time_metrics,
        "case_time_metrics.csv": tables.case_time_metrics,
        "extrema.csv": tables.extrema,
        "percentiles.csv": tables.percentiles,
    }
    for filename, rows in mappings.items():
        _write_csv_atomic(output_dir / filename, rows)
    payload = {
        "summary": tables.summary,
        "files": list(mappings),
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output_dir / "summary.json")
