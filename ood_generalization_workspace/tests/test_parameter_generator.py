from __future__ import annotations

import copy
import csv
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from ood_generalization_workspace.parameter_generator import (
    PARAMETER_GROUPS,
    TRAINING_BOUNDS,
    GenerationConfig,
    generate_samples,
    validate_samples,
    write_dataset_artifacts,
)


TEST_TEMP_ROOT = Path(__file__).parents[1].resolve()


def outside_parameters(sample: dict) -> list[str]:
    outside = []
    for group, names in PARAMETER_GROUPS.items():
        for name in names:
            lower, upper = TRAINING_BOUNDS[group][name]
            value = sample[group][name]
            if value < lower or value > upper:
                outside.append(name)
    return outside


class ParameterGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_and_has_expected_groups(self):
        config = GenerationConfig()
        first = generate_samples(config)
        second = generate_samples(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 150)
        self.assertEqual(first[0]["case_id"], "Case_1001")
        self.assertEqual(first[-1]["case_id"], "Case_1150")
        self.assertEqual(
            Counter(sample["ood"]["group"] for sample in first),
            {
                "geometry_ood": 50,
                "loading_ood": 50,
                "material_ood": 50,
            },
        )

    def test_each_case_has_exactly_one_outside_parameter_and_safe_geometry(self):
        config = GenerationConfig()
        samples = generate_samples(config)
        summary = validate_samples(samples, config)

        self.assertEqual(summary["valid_cases"], 150)
        for sample in samples:
            self.assertEqual(outside_parameters(sample), [sample["ood"]["parameter"]])
            self.assertIn(sample["ood"]["side"], {"lower", "upper"})
            self.assertGreater(sample["ood"]["normalized_ood_distance"], 0.0)
            self.assertLessEqual(sample["ood"]["normalized_ood_distance"], 0.1)
            safety_margin = (
                sample["geometry"]["sy"]
                - sample["geometry"]["b2"]
                - 2 * sample["loading"]["A"]
            )
            self.assertGreaterEqual(safety_margin, 10.0)

    def test_target_parameters_and_sides_are_balanced(self):
        samples = generate_samples(GenerationConfig())
        counts: dict[str, Counter] = defaultdict(Counter)
        side_counts: dict[str, Counter] = defaultdict(Counter)
        for sample in samples:
            group = sample["ood"]["group"]
            key = (sample["ood"]["parameter"], sample["ood"]["side"])
            counts[group][key] += 1
            side_counts[group][sample["ood"]["side"]] += 1

        for group_counts in counts.values():
            self.assertLessEqual(max(group_counts.values()) - min(group_counts.values()), 1)
        for group_side_counts in side_counts.values():
            self.assertLessEqual(
                max(group_side_counts.values()) - min(group_side_counts.values()), 1
            )

    def test_validation_rejects_a_second_outside_parameter(self):
        config = GenerationConfig()
        samples = generate_samples(config)
        broken = copy.deepcopy(samples)
        broken[0]["material"]["mu"] = 3500.0

        with self.assertRaisesRegex(ValueError, "exactly one outside parameter"):
            validate_samples(broken, config)

    def test_artifacts_round_trip_and_refuse_unrequested_overwrite(self):
        config = GenerationConfig()
        samples = generate_samples(config)
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            paths = write_dataset_artifacts(
                root, samples, config, overwrite=False
            )
            payload = json.loads(paths["parameters"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            with paths["audit"].open("r", encoding="utf-8-sig", newline="") as stream:
                audit_rows = list(csv.DictReader(stream))

            self.assertEqual(len(payload["parameters_list"]), 150)
            self.assertEqual(summary["validation"]["valid_cases"], 150)
            self.assertEqual(len(audit_rows), 150)
            self.assertEqual(audit_rows[0]["case_id"], "Case_1001")
            with self.assertRaises(FileExistsError):
                write_dataset_artifacts(root, samples, config, overwrite=False)

    def test_artifact_write_validates_samples_before_publishing(self):
        config = GenerationConfig()
        samples = generate_samples(config)
        samples[0]["geometry"]["sy"] = 1.0
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                write_dataset_artifacts(root, samples, config, overwrite=False)
            self.assertFalse((root / "4_Combined_Master_Dataset.json").exists())


if __name__ == "__main__":
    unittest.main()
