from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import EvaluationContext, attach_fields, predict_next
from .metrics import NormalizedMSEAccumulator, case_relative_threshold, compute_array_metrics
from .prediction_store import read_prediction


FIELD_NAMES = ("p", "T")


@dataclass
class TemporalComparison:
    case_id: str
    source_mode: str
    time_indices: np.ndarray
    time_steps: np.ndarray
    positions: np.ndarray
    velocity: np.ndarray
    face: np.ndarray
    region: np.ndarray
    truth: np.ndarray
    predictions: dict[str, np.ndarray]
    output_std: dict[str, np.ndarray]

    def validate(self) -> None:
        frame_count, node_count, field_count = self.truth.shape
        if field_count != len(FIELD_NAMES):
            raise ValueError("truth must have shape [frames, nodes, 2].")
        if self.time_indices.shape != (frame_count,):
            raise ValueError("time_indices shape mismatch.")
        if self.time_steps.shape != (frame_count,):
            raise ValueError("time_steps shape mismatch.")
        if self.positions.shape != (frame_count, node_count, 2):
            raise ValueError("positions shape mismatch.")
        if self.velocity.shape != (frame_count, node_count, 2):
            raise ValueError("velocity shape mismatch.")
        if self.region.shape != (node_count,):
            raise ValueError("region shape mismatch.")
        if self.face.ndim != 2 or self.face.shape[0] != 3:
            raise ValueError("face must have shape [3, F].")
        for model_name, values in self.predictions.items():
            if values.shape != self.truth.shape:
                raise ValueError(f"Prediction shape mismatch for {model_name}.")
            if np.asarray(self.output_std[model_name]).shape != (2,):
                raise ValueError(f"output_std shape mismatch for {model_name}.")


def _read_source_truth(context: EvaluationContext, case_id: str):
    source = context.dataset._resolve_file(case_id)
    with h5py.File(source, "r") as handle:
        times = handle["time_steps"][:].astype(np.float64)
        truth = np.stack(
            [handle[f"fields/{name}"][:] for name in FIELD_NAMES], axis=-1
        ).astype(np.float32)
    return times, truth


def _resolve_steps(start_index: int, steps: int | None, frame_count: int) -> int:
    available = frame_count - 1 - start_index
    if start_index < 0 or available < 1:
        raise ValueError("START_INDEX must leave at least one prediction step.")
    resolved = available if steps is None else min(int(steps), available)
    if resolved < 1:
        raise ValueError("STEPS must be at least 1.")
    return resolved


def load_saved_one_step_sequence(
    context: EvaluationContext,
    prediction_root: Path,
    case_id: str,
    start_index: int,
    steps: int | None,
) -> TemporalComparison:
    times, source_truth = _read_source_truth(context, case_id)
    resolved_steps = _resolve_steps(start_index, steps, len(times))
    target_indices = np.arange(
        start_index + 1, start_index + resolved_steps + 1, dtype=np.int64
    )
    initial_mesh = context.dataset.get_mesh_at_time(case_id, float(times[start_index]))
    positions = [initial_mesh.pos.numpy()]
    velocity = [initial_mesh.mesh_velocity.numpy()]
    predictions: dict[str, np.ndarray] = {}
    output_std: dict[str, np.ndarray] = {}
    shared_face = initial_mesh.face.numpy()
    shared_region = initial_mesh.mesh_region.numpy()
    target_positions = None
    target_velocity = None
    for model_name in context.model_names:
        stored = read_prediction(Path(prediction_root) / model_name / f"{case_id}.h5")
        if stored.model_name != model_name or stored.case_id != case_id:
            raise ValueError("Saved prediction metadata does not match the request.")
        if stored.checkpoint_sha256 != context.checkpoint_hashes[model_name]:
            raise ValueError(
                f"Saved {model_name} prediction was produced by a different checkpoint."
            )
        index_to_offset = {
            int(time_index): offset
            for offset, time_index in enumerate(stored.time_indices)
        }
        missing = [index for index in target_indices if int(index) not in index_to_offset]
        if missing:
            raise ValueError(f"Saved prediction misses target indices: {missing[:10]}")
        offsets = np.asarray([index_to_offset[int(index)] for index in target_indices])
        if target_positions is None:
            target_positions = stored.positions[offsets]
            target_velocity = stored.velocity[offsets]
            if not np.allclose(source_truth[target_indices], stored.truth[offsets]):
                raise ValueError("Saved prediction truth differs from the source HDF5.")
        else:
            if not np.allclose(target_positions, stored.positions[offsets]):
                raise ValueError("Saved model predictions use different mesh positions.")
            if not np.allclose(source_truth[target_indices], stored.truth[offsets]):
                raise ValueError("Saved model predictions use different truth fields.")
        predictions[model_name] = np.concatenate(
            [source_truth[start_index : start_index + 1], stored.prediction[offsets]],
            axis=0,
        )
        output_std[model_name] = stored.output_std
    positions.extend(list(target_positions))
    velocity.extend(list(target_velocity))
    result = TemporalComparison(
        case_id=case_id,
        source_mode="saved_one_step",
        time_indices=np.concatenate([[start_index], target_indices]),
        time_steps=times[np.concatenate([[start_index], target_indices])],
        positions=np.stack(positions),
        velocity=np.stack(velocity),
        face=shared_face,
        region=shared_region,
        truth=source_truth[np.concatenate([[start_index], target_indices])],
        predictions=predictions,
        output_std=output_std,
    )
    result.validate()
    return result


@torch.no_grad()
def rollout_sequence(
    context: EvaluationContext,
    case_id: str,
    start_index: int,
    steps: int | None,
) -> TemporalComparison:
    times, source_truth = _read_source_truth(context, case_id)
    resolved_steps = _resolve_steps(start_index, steps, len(times))
    frame_indices = np.arange(
        start_index, start_index + resolved_steps + 1, dtype=np.int64
    )
    loaded_models = {
        name: context.get_model(name) for name in context.model_names
    }
    current_fields = {
        name: torch.as_tensor(source_truth[start_index], dtype=torch.float32).to(
            context.device
        )
        for name in context.model_names
    }
    prediction_frames = {
        name: [source_truth[start_index].copy()] for name in context.model_names
    }
    positions = []
    velocity = []
    face = None
    region = None
    for offset, time_index in enumerate(frame_indices):
        output_mesh = context.dataset.get_mesh_at_time(
            case_id, float(times[time_index])
        )
        positions.append(output_mesh.pos.numpy())
        velocity.append(output_mesh.mesh_velocity.numpy())
        if face is None:
            face = output_mesh.face.numpy()
            region = output_mesh.mesh_region.numpy()
        if offset == resolved_steps:
            continue
        for model_name, loaded in loaded_models.items():
            input_mesh = context.dataset.get_mesh_at_time(
                case_id, float(times[time_index])
            )
            graph = attach_fields(input_mesh, current_fields[model_name].cpu())
            next_fields = predict_next(
                model_name, loaded, graph, context.graph_transform
            )
            current_fields[model_name] = next_fields
            prediction_frames[model_name].append(
                next_fields.detach().cpu().numpy().astype(np.float32, copy=False)
            )
    result = TemporalComparison(
        case_id=case_id,
        source_mode="rollout",
        time_indices=frame_indices,
        time_steps=times[frame_indices],
        positions=np.stack(positions),
        velocity=np.stack(velocity),
        face=np.asarray(face),
        region=np.asarray(region),
        truth=source_truth[frame_indices],
        predictions={name: np.stack(values) for name, values in prediction_frames.items()},
        output_std={
            name: loaded.output_std.numpy() for name, loaded in loaded_models.items()
        },
    )
    result.validate()
    return result


def build_step_metric_rows(
    sequence: TemporalComparison, threshold_ratio: float
) -> list[dict]:
    thresholds = [
        case_relative_threshold(sequence.truth[..., index], threshold_ratio)
        for index in range(len(FIELD_NAMES))
    ]
    rows = []
    for model_name, prediction in sequence.predictions.items():
        for frame_offset, time_index in enumerate(sequence.time_indices):
            row = {
                "model": model_name,
                "case_id": sequence.case_id,
                "source_mode": sequence.source_mode,
                "frame_offset": frame_offset,
                "rollout_horizon": frame_offset,
                "time_index": int(time_index),
                "physical_time": float(sequence.time_steps[frame_offset]),
            }
            for field_index, field_name in enumerate(FIELD_NAMES):
                metrics = compute_array_metrics(
                    prediction[frame_offset, :, field_index],
                    sequence.truth[frame_offset, :, field_index],
                    thresholds[field_index],
                )
                row.update({f"{field_name}_{key}": value for key, value in metrics.items()})
                row[f"{field_name}_relative_threshold"] = thresholds[field_index]
            normalized = NormalizedMSEAccumulator(sequence.output_std[model_name])
            normalized.update(prediction[frame_offset], sequence.truth[frame_offset])
            row["normalized_mse"] = normalized.value
            rows.append(row)
    return rows


def write_step_metrics(rows: list[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)
