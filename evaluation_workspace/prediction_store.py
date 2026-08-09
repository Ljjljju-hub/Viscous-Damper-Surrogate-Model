from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


FIELD_NAMES = ("p", "T")


@dataclass
class PredictionCase:
    case_id: str
    model_name: str
    checkpoint_path: str
    checkpoint_sha256: str
    time_indices: np.ndarray
    time_steps: np.ndarray
    positions: np.ndarray
    velocity: np.ndarray
    face: np.ndarray
    region: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray

    def validate(self) -> None:
        time_count = len(self.time_indices)
        if self.time_steps.shape != (time_count,):
            raise ValueError("time_steps shape does not match time_indices.")
        if self.truth.ndim != 3 or self.truth.shape[-1] != len(FIELD_NAMES):
            raise ValueError("truth must have shape [K, N, 2].")
        if self.prediction.shape != self.truth.shape:
            raise ValueError("prediction shape must match truth.")
        expected_vector_shape = self.truth.shape[:2] + (2,)
        if self.positions.shape != expected_vector_shape:
            raise ValueError("positions must have shape [K, N, 2].")
        if self.velocity.shape != expected_vector_shape:
            raise ValueError("velocity must have shape [K, N, 2].")
        if self.region.shape != (self.truth.shape[1],):
            raise ValueError("region must have shape [N].")
        if self.face.ndim != 2 or self.face.shape[0] != 3:
            raise ValueError("face must have shape [3, F].")
        if np.asarray(self.output_mean).shape != (2,):
            raise ValueError("output_mean must have shape [2].")
        if np.asarray(self.output_std).shape != (2,):
            raise ValueError("output_std must have shape [2].")
        for name, values in (
            ("positions", self.positions),
            ("velocity", self.velocity),
            ("truth", self.truth),
            ("prediction", self.prediction),
            ("output_mean", self.output_mean),
            ("output_std", self.output_std),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values.")
        if np.any(np.asarray(self.output_std) <= 0.0):
            raise ValueError("output_std must be positive.")


def _create_dataset(group, name: str, values, dtype=None) -> None:
    array = np.asarray(values, dtype=dtype)
    group.create_dataset(name, data=array, compression="gzip")


def write_prediction_atomic(path: Path, data: PredictionCase) -> None:
    data.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.h5")
    partial.unlink(missing_ok=True)
    try:
        with h5py.File(partial, "w") as handle:
            handle.attrs.update(
                {
                    "model_name": data.model_name,
                    "case_id": data.case_id,
                    "checkpoint_path": data.checkpoint_path,
                    "checkpoint_sha256": data.checkpoint_sha256,
                    "prediction_mode": "one_step",
                    "complete": False,
                }
            )
            _create_dataset(handle, "time_indices", data.time_indices, np.int64)
            _create_dataset(handle, "time_steps", data.time_steps, np.float64)
            mesh = handle.create_group("mesh")
            _create_dataset(mesh, "positions", data.positions, np.float32)
            _create_dataset(mesh, "velocity", data.velocity, np.float32)
            _create_dataset(mesh, "face", data.face, np.int64)
            _create_dataset(mesh, "region", data.region, np.int64)
            truth = handle.create_group("truth")
            prediction = handle.create_group("prediction")
            for field_index, field_name in enumerate(FIELD_NAMES):
                _create_dataset(
                    truth, field_name, data.truth[..., field_index], np.float32
                )
                _create_dataset(
                    prediction,
                    field_name,
                    data.prediction[..., field_index],
                    np.float32,
                )
            normalization = handle.create_group("normalization")
            _create_dataset(
                normalization, "output_mean", data.output_mean, np.float64
            )
            _create_dataset(normalization, "output_std", data.output_std, np.float64)
            handle.attrs["complete"] = True
            handle.flush()
        if not prediction_is_reusable(
            partial,
            model_name=data.model_name,
            case_id=data.case_id,
            checkpoint_sha256=data.checkpoint_sha256,
        ):
            raise RuntimeError(f"Prediction HDF5 validation failed: {partial}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def read_prediction(path: Path) -> PredictionCase:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        truth = np.stack([handle[f"truth/{name}"][:] for name in FIELD_NAMES], axis=-1)
        prediction = np.stack(
            [handle[f"prediction/{name}"][:] for name in FIELD_NAMES], axis=-1
        )
        result = PredictionCase(
            case_id=str(handle.attrs["case_id"]),
            model_name=str(handle.attrs["model_name"]),
            checkpoint_path=str(handle.attrs["checkpoint_path"]),
            checkpoint_sha256=str(handle.attrs["checkpoint_sha256"]),
            time_indices=handle["time_indices"][:],
            time_steps=handle["time_steps"][:],
            positions=handle["mesh/positions"][:],
            velocity=handle["mesh/velocity"][:],
            face=handle["mesh/face"][:],
            region=handle["mesh/region"][:],
            truth=truth,
            prediction=prediction,
            output_mean=handle["normalization/output_mean"][:],
            output_std=handle["normalization/output_std"][:],
        )
    result.validate()
    return result


def prediction_is_reusable(
    path: Path,
    *,
    model_name: str,
    case_id: str,
    checkpoint_sha256: str,
    expected_time_count: int | None = None,
) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                return False
            if str(handle.attrs.get("model_name", "")) != model_name:
                return False
            if str(handle.attrs.get("case_id", "")) != case_id:
                return False
            if str(handle.attrs.get("checkpoint_sha256", "")) != checkpoint_sha256:
                return False
            time_count = len(handle["time_indices"])
            if expected_time_count is not None and time_count != expected_time_count:
                return False
            node_count = len(handle["mesh/region"])
            required = [
                handle["time_steps"].shape == (time_count,),
                handle["mesh/positions"].shape == (time_count, node_count, 2),
                handle["mesh/velocity"].shape == (time_count, node_count, 2),
                handle["truth/p"].shape == (time_count, node_count),
                handle["truth/T"].shape == (time_count, node_count),
                handle["prediction/p"].shape == (time_count, node_count),
                handle["prediction/T"].shape == (time_count, node_count),
                handle["normalization/output_std"].shape == (2,),
            ]
            return time_count > 0 and node_count > 0 and all(required)
    except (OSError, KeyError, ValueError):
        return False
