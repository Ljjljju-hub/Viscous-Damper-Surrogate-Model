import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation_workspace.prediction_store import PredictionCase, write_prediction_atomic
from evaluation_workspace.test_pipeline import analyze_prediction_directory


class TestPipelineTests(unittest.TestCase):
    def make_case(self, model, case_id, scale):
        truth = np.zeros((2, 3, 2), dtype=np.float32)
        truth[..., 0] = np.array([[10, 20, 30], [20, 30, 40]])
        truth[..., 1] = 300.0
        prediction = truth + np.array([scale, scale / 10], dtype=np.float32)
        return PredictionCase(
            case_id=case_id,
            model_name=model,
            checkpoint_path="best.pt",
            checkpoint_sha256=f"{model}-hash",
            time_indices=np.array([1, 2]),
            time_steps=np.array([0.1, 0.2]),
            positions=np.zeros((2, 3, 2), dtype=np.float32),
            velocity=np.zeros((2, 3, 2), dtype=np.float32),
            face=np.array([[0], [1], [2]], dtype=np.int64),
            region=np.array([0, 1, 2]),
            truth=truth,
            prediction=prediction,
            output_mean=np.zeros(2),
            output_std=np.array([2.0, 1.0]),
        )

    def test_hierarchical_row_counts_and_global_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("meshgraphnet", "transolver"):
                for case_index, scale in enumerate((1.0, 3.0), 1):
                    data = self.make_case(model, f"Case_{case_index:04d}", scale)
                    write_prediction_atomic(
                        root / model / f"{data.case_id}.h5", data
                    )
            tables = analyze_prediction_directory(root)

        self.assertEqual(len(tables.summary), 2)
        self.assertEqual(len(tables.case_metrics), 4)
        self.assertEqual(len(tables.time_metrics), 4)
        self.assertEqual(len(tables.case_time_metrics), 8)
        expected_pressure_rmse = np.sqrt((1.0**2 + 3.0**2) / 2.0)
        self.assertAlmostEqual(
            tables.summary[0]["p_rmse"], expected_pressure_rmse
        )
        global_pressure_extrema = [
            row
            for row in tables.extrema
            if row["scope"] == "global"
            and row["model"] == "meshgraphnet"
            and row["field"] == "p"
            and row["metric_type"] == "absolute"
        ]
        self.assertEqual(global_pressure_extrema[0]["case_id"], "Case_0002")


if __name__ == "__main__":
    unittest.main()
