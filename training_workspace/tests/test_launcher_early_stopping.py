import unittest
from types import SimpleNamespace

from training_workspace.run_scale_study import (
    early_stopping_command_args,
    early_stopping_request,
)


class LauncherEarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            early_stopping_patience=10,
            early_stopping_min_relative_improvement=0.002,
        )

    def test_request_records_early_stopping_configuration(self):
        self.assertEqual(
            early_stopping_request(self.args),
            {
                "early_stopping_patience": 10,
                "early_stopping_min_relative_improvement": 0.002,
            },
        )

    def test_worker_command_receives_early_stopping_configuration(self):
        self.assertEqual(
            early_stopping_command_args(self.args),
            [
                "--early-stopping-patience",
                "10",
                "--early-stopping-min-relative-improvement",
                "0.002",
            ],
        )


if __name__ == "__main__":
    unittest.main()
