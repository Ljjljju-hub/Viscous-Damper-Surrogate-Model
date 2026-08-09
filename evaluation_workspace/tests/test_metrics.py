import unittest

import numpy as np

from evaluation_workspace.metrics import (
    MetricAccumulator,
    NormalizedMSEAccumulator,
    case_relative_threshold,
    compute_array_metrics,
)


class MetricTests(unittest.TestCase):
    def test_global_rmse_pools_squared_errors_before_root(self):
        accumulator = MetricAccumulator("p")
        accumulator.update(np.array([0.0, 2.0]), np.array([0.0, 0.0]), 0.0)
        accumulator.update(np.array([6.0]), np.array([0.0]), 0.0)
        self.assertAlmostEqual(
            accumulator.finalize()["rmse"], np.sqrt(40.0 / 3.0)
        )

    def test_relative_error_excludes_values_below_case_threshold(self):
        result = compute_array_metrics(
            prediction=np.array([10.0, 2.0, 0.5]),
            truth=np.array([8.0, 1.0, 0.0]),
            relative_threshold=0.08,
        )
        self.assertEqual(result["relative_valid_count"], 2)
        self.assertEqual(result["relative_excluded_count"], 1)
        self.assertAlmostEqual(result["max_relative_error_percent"], 100.0)

    def test_case_threshold_uses_all_case_values(self):
        truth = np.array([[0.0, -20.0], [5.0, 10.0]])
        self.assertAlmostEqual(case_relative_threshold(truth, 0.01), 0.2)

    def test_normalized_mse_averages_both_fields(self):
        accumulator = NormalizedMSEAccumulator(np.array([2.0, 4.0]))
        truth = np.zeros((2, 2))
        prediction = np.array([[2.0, 4.0], [4.0, 0.0]])
        accumulator.update(prediction, truth)
        self.assertAlmostEqual(accumulator.value, (1.0 + 1.0 + 4.0) / 4.0)


if __name__ == "__main__":
    unittest.main()
