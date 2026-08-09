import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation_workspace.prediction_store import (
    PredictionCase,
    prediction_is_reusable,
    read_prediction,
    write_prediction_atomic,
)


class PredictionStoreTests(unittest.TestCase):
    def make_case(self):
        return PredictionCase(
            case_id="Case_0001",
            model_name="meshgraphnet",
            checkpoint_path="best.pt",
            checkpoint_sha256="abc123",
            time_indices=np.array([1, 2], dtype=np.int64),
            time_steps=np.array([0.1, 0.2]),
            positions=np.zeros((2, 3, 2), dtype=np.float32),
            velocity=np.ones((2, 3, 2), dtype=np.float32),
            face=np.array([[0], [1], [2]], dtype=np.int64),
            region=np.array([0, 1, 2], dtype=np.int64),
            truth=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
            prediction=np.arange(12, dtype=np.float32).reshape(2, 3, 2) + 1,
            output_mean=np.array([0.5, 1.5]),
            output_std=np.array([2.0, 4.0]),
        )

    def test_atomic_round_trip_and_reuse_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Case_0001.h5"
            source = self.make_case()
            write_prediction_atomic(path, source)
            restored = read_prediction(path)

            self.assertEqual(restored.case_id, source.case_id)
            np.testing.assert_allclose(restored.prediction, source.prediction)
            np.testing.assert_allclose(restored.output_std, source.output_std)
            self.assertTrue(
                prediction_is_reusable(
                    path,
                    model_name="meshgraphnet",
                    case_id="Case_0001",
                    checkpoint_sha256="abc123",
                    expected_time_count=2,
                )
            )
            self.assertFalse(
                prediction_is_reusable(
                    path,
                    model_name="meshgraphnet",
                    case_id="Case_0001",
                    checkpoint_sha256="changed",
                )
            )
            self.assertFalse(
                prediction_is_reusable(
                    path,
                    model_name="meshgraphnet",
                    case_id="Case_0001",
                    checkpoint_sha256="abc123",
                    expected_time_count=150,
                )
            )


if __name__ == "__main__":
    unittest.main()
