from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def case_relative_threshold(truth: np.ndarray, ratio: float) -> float:
    if not 0.0 <= ratio < 1.0:
        raise ValueError("relative-error threshold ratio must be in [0, 1).")
    values = np.asarray(truth, dtype=np.float64)
    if values.size == 0:
        raise ValueError("truth must contain at least one value.")
    return float(np.max(np.abs(values)) * ratio)


def relative_error_mask(truth: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(truth)
    return (np.abs(values) >= float(threshold)) & (np.abs(values) > 0.0)


@dataclass
class MetricAccumulator:
    field_name: str
    collect_absolute_errors: bool = False
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    value_count: int = 0
    relative_valid_count: int = 0
    relative_excluded_count: int = 0
    max_absolute_error: float = -math.inf
    max_relative_error_percent: float = -math.inf
    _absolute_error_chunks: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        prediction: np.ndarray,
        truth: np.ndarray,
        relative_threshold: float,
    ) -> None:
        prediction_values = np.asarray(prediction, dtype=np.float64)
        truth_values = np.asarray(truth, dtype=np.float64)
        if prediction_values.shape != truth_values.shape:
            raise ValueError(
                f"prediction/truth shape mismatch: "
                f"{prediction_values.shape} != {truth_values.shape}"
            )
        error = prediction_values - truth_values
        absolute_error = np.abs(error)
        self.sum_squared_error += float(np.square(error).sum(dtype=np.float64))
        self.sum_absolute_error += float(absolute_error.sum(dtype=np.float64))
        self.value_count += int(error.size)
        if error.size:
            self.max_absolute_error = max(
                self.max_absolute_error, float(absolute_error.max())
            )
        if self.collect_absolute_errors:
            self._absolute_error_chunks.append(
                absolute_error.astype(np.float32, copy=False).reshape(-1).copy()
            )

        valid = relative_error_mask(truth_values, relative_threshold)
        valid_count = int(valid.sum())
        self.relative_valid_count += valid_count
        self.relative_excluded_count += int(valid.size - valid_count)
        if valid_count:
            relative = absolute_error[valid] / np.abs(truth_values[valid]) * 100.0
            self.max_relative_error_percent = max(
                self.max_relative_error_percent, float(relative.max())
            )

    def finalize(self) -> dict[str, float | int]:
        if self.value_count == 0:
            raise ValueError(f"No values accumulated for field {self.field_name!r}.")
        result: dict[str, float | int] = {
            "count": self.value_count,
            "rmse": math.sqrt(self.sum_squared_error / self.value_count),
            "mae": self.sum_absolute_error / self.value_count,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error_percent": (
                self.max_relative_error_percent
                if math.isfinite(self.max_relative_error_percent)
                else math.nan
            ),
            "relative_valid_count": self.relative_valid_count,
            "relative_excluded_count": self.relative_excluded_count,
        }
        if self.collect_absolute_errors:
            errors = np.concatenate(self._absolute_error_chunks)
            p95, p99 = np.percentile(errors, [95.0, 99.0])
            result.update(
                {
                    "p95_absolute_error": float(p95),
                    "p99_absolute_error": float(p99),
                }
            )
        return result


def compute_array_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    relative_threshold: float,
    *,
    collect_absolute_errors: bool = False,
) -> dict[str, float | int]:
    accumulator = MetricAccumulator(
        "field", collect_absolute_errors=collect_absolute_errors
    )
    accumulator.update(prediction, truth, relative_threshold)
    return accumulator.finalize()


@dataclass
class NormalizedMSEAccumulator:
    output_std: np.ndarray
    sum_squared_error: float = 0.0
    value_count: int = 0

    def __post_init__(self) -> None:
        self.output_std = np.asarray(self.output_std, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(self.output_std)) or np.any(self.output_std <= 0.0):
            raise ValueError("output_std must contain positive finite values.")

    def update(self, prediction: np.ndarray, truth: np.ndarray) -> None:
        prediction_values = np.asarray(prediction, dtype=np.float64)
        truth_values = np.asarray(truth, dtype=np.float64)
        if prediction_values.shape != truth_values.shape:
            raise ValueError("prediction/truth shape mismatch.")
        if prediction_values.shape[-1] != self.output_std.size:
            raise ValueError("Last dimension must match output_std.")
        scaled_error = (prediction_values - truth_values) / self.output_std
        self.sum_squared_error += float(
            np.square(scaled_error).sum(dtype=np.float64)
        )
        self.value_count += int(scaled_error.size)

    @property
    def value(self) -> float:
        if self.value_count == 0:
            raise ValueError("No values accumulated for normalized MSE.")
        return self.sum_squared_error / self.value_count
