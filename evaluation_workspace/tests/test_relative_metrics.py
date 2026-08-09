from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch_geometric.data import Data

from evaluation_workspace.calculate_relative_metrics import (
    calculate_one_step_case,
    calculate_rollout_case,
    rollout_prediction_path,
)
from evaluation_workspace.prediction_store import PredictionCase, write_prediction_atomic
from evaluation_workspace.relative_metrics import (
    RelativeMetricAccumulator,
    compute_relative_field_metrics,
    temperature_rise,
)


class TemperatureRiseTests(unittest.TestCase):
    def test_temperature_rise_uses_case_initial_temperature(self):
        temperature = np.array([[300.0, 302.0], [303.0, 306.0]])
        initial = np.array([299.0, 301.0])

        result = temperature_rise(temperature, initial)

        np.testing.assert_allclose(result, [[1.0, 1.0], [4.0, 5.0]])

    def test_temperature_rise_rejects_node_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "initial_temperature"):
            temperature_rise(np.zeros((2, 3)), np.zeros(2))


class RelativeFieldMetricTests(unittest.TestCase):
    def test_computes_global_relative_rmse_and_point_percentiles(self):
        truth = np.array([1.0, 2.0, 4.0, 8.0])
        prediction = np.array([2.0, 4.0, 6.0, 12.0])

        result = compute_relative_field_metrics(
            prediction, truth, threshold_ratio=0.2
        )

        expected_rmse = np.sqrt(np.mean(np.square(prediction - truth)))
        expected_gt_rms = np.sqrt(np.mean(np.square(truth)))
        expected_relative = expected_rmse / expected_gt_rms * 100.0
        valid_relative = np.array([100.0, 50.0, 50.0])
        self.assertAlmostEqual(result["absolute_rmse"], expected_rmse)
        self.assertAlmostEqual(result["gt_rms"], expected_gt_rms)
        self.assertAlmostEqual(
            result["relative_rmse_percent"], expected_relative
        )
        self.assertEqual(result["point_relative_valid_count"], 3)
        self.assertEqual(result["point_relative_excluded_count"], 1)
        self.assertAlmostEqual(
            result["point_relative_p50_percent"],
            float(np.percentile(valid_relative, 50.0)),
        )
        self.assertAlmostEqual(
            result["point_relative_p95_percent"],
            float(np.percentile(valid_relative, 95.0)),
        )
        self.assertAlmostEqual(
            result["point_relative_p99_percent"],
            float(np.percentile(valid_relative, 99.0)),
        )
        self.assertEqual(result["point_relative_max_percent"], 100.0)

    def test_accumulator_combines_cases_with_per_case_thresholds(self):
        accumulator = RelativeMetricAccumulator("delta_T")
        accumulator.update(
            prediction=np.array([0.0, 3.0]),
            truth=np.array([0.1, 2.0]),
            threshold=0.2,
        )
        accumulator.update(
            prediction=np.array([11.0, 22.0]),
            truth=np.array([10.0, 20.0]),
            threshold=2.0,
        )

        result = accumulator.finalize()

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["point_relative_valid_count"], 3)
        self.assertEqual(result["point_relative_excluded_count"], 1)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compute_relative_field_metrics(
                np.zeros(2), np.zeros(3), threshold_ratio=0.01
            )

    def test_rejects_all_zero_truth_for_relative_rmse(self):
        with self.assertRaisesRegex(ValueError, "GT energy"):
            compute_relative_field_metrics(
                np.ones(3), np.zeros(3), threshold_ratio=0.01
            )


class SavedPredictionMetricTests(unittest.TestCase):
    def _temporary_directory(self):
        return TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def setUp(self):
        self.initial_fields = np.array(
            [[10.0, 300.0], [20.0, 310.0]], dtype=np.float32
        )
        self.truth = np.array(
            [
                [[11.0, 301.0], [22.0, 312.0]],
                [[13.0, 304.0], [25.0, 315.0]],
            ],
            dtype=np.float32,
        )
        self.prediction = self.truth + np.array([1.0, 0.5], dtype=np.float32)

    def _write_one_step(self, path: Path) -> None:
        write_prediction_atomic(
            path,
            PredictionCase(
                case_id="Case_0001",
                model_name="transolver",
                checkpoint_path="best.pt",
                checkpoint_sha256="abc",
                time_indices=np.array([1, 2]),
                time_steps=np.array([0.1, 0.2]),
                positions=np.zeros((2, 2, 2), dtype=np.float32),
                velocity=np.zeros((2, 2, 2), dtype=np.float32),
                face=np.array([[0], [1], [0]], dtype=np.int64),
                region=np.zeros(2, dtype=np.int64),
                truth=self.truth,
                prediction=self.prediction,
                output_mean=np.zeros(2),
                output_std=np.ones(2),
            ),
        )

    def test_one_step_uses_gt_initial_temperature_for_delta_T(self):
        with self._temporary_directory() as temporary_directory:
            path = Path(temporary_directory) / "Case_0001.h5"
            self._write_one_step(path)

            result = calculate_one_step_case(
                path, self.initial_fields, threshold_ratio=0.01
            )

        self.assertAlmostEqual(
            result["T"]["absolute_rmse"],
            result["delta_T"]["absolute_rmse"],
        )
        expected_delta_truth = self.truth[..., 1] - self.initial_fields[:, 1]
        self.assertAlmostEqual(
            result["delta_T"]["gt_rms"],
            float(np.sqrt(np.mean(np.square(expected_delta_truth)))),
        )

    def test_rollout_uses_gt_initial_temperature_for_delta_T(self):
        with self._temporary_directory() as temporary_directory:
            path = Path(temporary_directory) / "Case_0001.pt"
            predicted_frames = [
                Data(x=torch.as_tensor(self.initial_fields)),
                *[
                    Data(
                        x=torch.cat(
                            [
                                torch.zeros((2, 1)),
                                torch.as_tensor(frame),
                            ],
                            dim=1,
                        )
                    )
                    for frame in self.prediction
                ],
            ]
            torch.save(
                {
                    "case_id": "Case_0001",
                    "field_names": ("p", "T"),
                    "meshes": predicted_frames,
                },
                path,
            )
            all_truth = np.concatenate(
                [self.initial_fields[None, ...], self.truth], axis=0
            )

            result = calculate_rollout_case(
                path, all_truth, threshold_ratio=0.01
            )

        self.assertAlmostEqual(
            result["T"]["absolute_rmse"],
            result["delta_T"]["absolute_rmse"],
        )
        self.assertAlmostEqual(
            result["delta_T"]["gt_rms"],
            float(
                np.sqrt(
                    np.mean(
                        np.square(self.truth[..., 1] - self.initial_fields[:, 1])
                    )
                )
            ),
        )

    def test_one_step_rejects_initial_node_mismatch(self):
        with self._temporary_directory() as temporary_directory:
            path = Path(temporary_directory) / "Case_0001.h5"
            self._write_one_step(path)

            with self.assertRaisesRegex(ValueError, "initial_fields"):
                calculate_one_step_case(
                    path, np.zeros((3, 2)), threshold_ratio=0.01
                )

    def test_rollout_path_does_not_repeat_model_directory(self):
        root = Path("runs/meshgraphnet/n0100/seed_42/rollouts")

        result = rollout_prediction_path(root, "Case_0866")

        self.assertEqual(result, root / "Case_0866.pt")


if __name__ == "__main__":
    unittest.main()
