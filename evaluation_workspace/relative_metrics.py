from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def temperature_rise(
    temperature: np.ndarray, initial_temperature: np.ndarray
) -> np.ndarray:
    values = np.asarray(temperature, dtype=np.float64)
    initial = np.asarray(initial_temperature, dtype=np.float64)
    if values.ndim < 1 or initial.ndim != 1 or values.shape[-1] != initial.shape[0]:
        raise ValueError(
            "initial_temperature must have shape [N] matching temperature nodes."
        )
    return values - initial


def _case_threshold(truth: np.ndarray, ratio: float) -> float:
    if not 0.0 <= ratio < 1.0:
        raise ValueError("threshold_ratio must be in [0, 1).")
    return float(np.max(np.abs(truth)) * ratio)


@dataclass
class RelativeMetricAccumulator:
    field_name: str
    sum_squared_error: float = 0.0
    sum_squared_truth: float = 0.0
    count: int = 0
    point_relative_valid_count: int = 0
    point_relative_excluded_count: int = 0
    _point_relative_chunks: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        prediction: np.ndarray,
        truth: np.ndarray,
        threshold: float,
    ) -> None:
        prediction_values = np.asarray(prediction, dtype=np.float64)
        truth_values = np.asarray(truth, dtype=np.float64)
        if prediction_values.shape != truth_values.shape:
            raise ValueError(
                f"prediction/truth shape mismatch: "
                f"{prediction_values.shape} != {truth_values.shape}"
            )
        if prediction_values.size == 0:
            raise ValueError("prediction and truth must not be empty.")
        if not np.isfinite(prediction_values).all() or not np.isfinite(truth_values).all():
            raise ValueError("prediction and truth must contain finite values.")

        error = prediction_values - truth_values
        self.sum_squared_error += float(np.square(error).sum(dtype=np.float64))
        self.sum_squared_truth += float(
            np.square(truth_values).sum(dtype=np.float64)
        )
        self.count += int(error.size)

        valid = (np.abs(truth_values) >= float(threshold)) & (
            np.abs(truth_values) > 0.0
        )
        valid_count = int(valid.sum())
        self.point_relative_valid_count += valid_count
        self.point_relative_excluded_count += int(valid.size - valid_count)
        if valid_count:
            relative = np.abs(error[valid]) / np.abs(truth_values[valid]) * 100.0
            self._point_relative_chunks.append(
                relative.astype(np.float32, copy=False).reshape(-1).copy()
            )

    def finalize(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError(f"No values accumulated for {self.field_name!r}.")
        if self.sum_squared_truth <= 0.0:
            raise ValueError(f"GT energy is zero for {self.field_name!r}.")
        if not self._point_relative_chunks:
            raise ValueError(
                f"No valid point-relative values for {self.field_name!r}."
            )

        point_relative = np.concatenate(self._point_relative_chunks)
        p50, p95, p99 = np.percentile(point_relative, [50.0, 95.0, 99.0])
        return {
            "count": self.count,
            "absolute_rmse": math.sqrt(self.sum_squared_error / self.count),
            "gt_rms": math.sqrt(self.sum_squared_truth / self.count),
            "relative_rmse_percent": math.sqrt(
                self.sum_squared_error / self.sum_squared_truth
            )
            * 100.0,
            "point_relative_valid_count": self.point_relative_valid_count,
            "point_relative_excluded_count": self.point_relative_excluded_count,
            "point_relative_p50_percent": float(p50),
            "point_relative_p95_percent": float(p95),
            "point_relative_p99_percent": float(p99),
            "point_relative_max_percent": float(point_relative.max()),
        }


def compute_relative_field_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    threshold_ratio: float,
) -> dict[str, float | int]:
    truth_values = np.asarray(truth, dtype=np.float64)
    if truth_values.size == 0:
        raise ValueError("truth must not be empty.")
    accumulator = RelativeMetricAccumulator("field")
    accumulator.update(
        prediction,
        truth_values,
        threshold=_case_threshold(truth_values, threshold_ratio),
    )
    return accumulator.finalize()
