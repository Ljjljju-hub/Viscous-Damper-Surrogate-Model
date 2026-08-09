import csv
import tempfile
import unittest
from pathlib import Path

from evaluation_workspace.representative_cases import select_representative_cases


class RepresentativeCaseTests(unittest.TestCase):
    def test_selects_min_and_max_of_each_case_level_metric(self):
        rows = [
            {
                "model": "meshgraphnet",
                "case_id": "Case_0001",
                "p_rmse": 2.0,
                "p_max_absolute_error": 8.0,
                "p_max_relative_error_percent": 20.0,
                "T_rmse": 0.2,
                "T_max_absolute_error": 0.8,
                "T_max_relative_error_percent": 2.0,
            },
            {
                "model": "meshgraphnet",
                "case_id": "Case_0002",
                "p_rmse": 5.0,
                "p_max_absolute_error": 6.0,
                "p_max_relative_error_percent": 30.0,
                "T_rmse": 0.1,
                "T_max_absolute_error": 1.8,
                "T_max_relative_error_percent": 1.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case_metrics.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            selected = select_representative_cases(path, ["meshgraphnet"])

        self.assertEqual(len(selected), 12)
        pressure_rmse_max = next(
            row
            for row in selected
            if row["field"] == "p"
            and row["metric"] == "rmse"
            and row["extreme"] == "max"
        )
        self.assertEqual(pressure_rmse_max["case_id"], "Case_0002")
        pressure_point_min = next(
            row
            for row in selected
            if row["field"] == "p"
            and row["metric"] == "max_absolute_error"
            and row["extreme"] == "min"
        )
        self.assertEqual(pressure_point_min["case_id"], "Case_0002")


if __name__ == "__main__":
    unittest.main()
