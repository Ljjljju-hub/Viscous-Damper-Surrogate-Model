from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ood_generalization_workspace.parameter_generator import (
    GenerationConfig,
    generate_samples,
    write_dataset_artifacts,
)
from ood_generalization_workspace.run_pipeline import (
    PipelineConfig,
    build_controller_command,
    prepare_workspace_parameters,
    validate_workspace_parameters,
)


TEST_TEMP_ROOT = Path(__file__).parents[1].resolve()


class PipelineTests(unittest.TestCase):
    def test_controller_command_targets_only_ood_workspace(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory).resolve()
            model = root / "standard_model.mph"
            config = PipelineConfig(
                workspace_root=root,
                model_path=model,
                dry_run=True,
                batch_size=7,
                cores=6,
                show_worker_window=False,
            )

            command = build_controller_command(config)

            self.assertEqual(command[0], config.python_executable)
            self.assertEqual(
                command[command.index("--workspace-root") + 1], str(root)
            )
            self.assertEqual(
                command[command.index("--model-path") + 1], str(model)
            )
            self.assertEqual(command[command.index("--batch-size") + 1], "7")
            self.assertIn("--dry-run", command)
            self.assertIn("--no-worker-window", command)

    def test_validation_rejects_invalid_existing_parameter_file(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            (root / "4_Combined_Master_Dataset.json").write_text(
                json.dumps({"parameters_list": []}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_workspace_parameters(root, GenerationConfig())

    def test_prepare_reuses_valid_existing_parameters_without_overwrite(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            generation = GenerationConfig()
            samples = generate_samples(generation)
            write_dataset_artifacts(root, samples, generation, overwrite=False)
            parameter_path = root / "4_Combined_Master_Dataset.json"
            original_text = parameter_path.read_text(encoding="utf-8")

            summary = prepare_workspace_parameters(
                root,
                generation,
                generate=True,
                overwrite=False,
            )

            self.assertEqual(summary["valid_cases"], 150)
            self.assertEqual(parameter_path.read_text(encoding="utf-8"), original_text)


if __name__ == "__main__":
    unittest.main()
