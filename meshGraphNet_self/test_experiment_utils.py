import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiment_utils import (
    capture_rng_state,
    restore_rng_state,
    select_manifest_cases,
)


class ExperimentUtilsTest(unittest.TestCase):
    def test_nested_manifest_selection(self):
        content = json.dumps(
            {
                "train_pool": ["Case_3", "Case_1", "Case_2"],
                "valid": ["Case_4"],
                "test": ["Case_5"],
            }
        )
        with patch.object(Path, "read_text", return_value=content):
            train, valid, test, _ = select_manifest_cases(
                Path("unused-split.json"), 2
            )
        self.assertEqual(train, ["Case_3", "Case_1"])
        self.assertEqual(valid, ["Case_4"])
        self.assertEqual(test, ["Case_5"])

    def test_rng_state_restores_all_generators(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        generator = torch.Generator().manual_seed(7)
        state = capture_rng_state(generator)
        expected = (
            random.random(),
            np.random.rand(),
            torch.rand(1),
            torch.rand(1, generator=generator),
        )
        restore_rng_state(state, generator)
        actual = (
            random.random(),
            np.random.rand(),
            torch.rand(1),
            torch.rand(1, generator=generator),
        )
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2])
        torch.testing.assert_close(expected[3], actual[3])


if __name__ == "__main__":
    unittest.main()
