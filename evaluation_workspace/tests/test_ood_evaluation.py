from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from evaluation_workspace.common import load_evaluation_context_from_cases
from evaluation_workspace.ood_evaluation import (
    OodCaseInventory,
    build_ood_inventory,
    write_ood_case_audit,
)
from evaluation_workspace.test_ood import main as ood_test_main


class OodInventoryTests(unittest.TestCase):
    def _temporary_directory(self):
        return TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def _write_hdf5(self, path: Path, *, include_temperature: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("time_steps", data=np.array([0.0, 0.1]))
            mesh = handle.create_group("mesh")
            mesh.create_dataset(
                "coordinates",
                data=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            )
            mesh.create_dataset(
                "connectivity", data=np.array([3, 0, 1, 2], dtype=np.int64)
            )
            fields = handle.create_group("fields")
            fields.create_dataset("p", data=np.ones((2, 3)))
            if include_temperature:
                fields.create_dataset("T", data=np.full((2, 3), 300.0))

    def _make_workspace(self, root: Path) -> Path:
        workspace = root / "ood"
        workspace.mkdir()
        parameters = [
            {"case_id": "Case_1001"},
            {"case_id": "Case_1002"},
            {"case_id": "Case_1003"},
        ]
        (workspace / "4_Combined_Master_Dataset.json").write_text(
            json.dumps({"parameters_list": parameters}), encoding="utf-8"
        )
        with (workspace / "parameter_audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("case_id", "ood_group", "ood_parameter", "ood_side"),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "case_id": item["case_id"],
                        "ood_group": "geometry_ood",
                        "ood_parameter": "r1",
                        "ood_side": "lower",
                    }
                    for item in parameters
                ]
            )
        (workspace / "failed_cases.json").write_text(
            json.dumps({"cases": {"Case_1002": {"source": "worker.log"}}}),
            encoding="utf-8",
        )
        self._write_hdf5(workspace / "comsol_hdf5" / "Case_1001.h5")
        self._write_hdf5(workspace / "comsol_hdf5" / "Case_1003.h5")
        return workspace

    def test_inventory_partitions_all_parameter_cases(self):
        with self._temporary_directory() as directory:
            workspace = self._make_workspace(Path(directory))

            inventory = build_ood_inventory(workspace)

        self.assertEqual(
            inventory.parameter_case_ids,
            ("Case_1001", "Case_1002", "Case_1003"),
        )
        self.assertEqual(inventory.valid_case_ids, ("Case_1001", "Case_1003"))
        self.assertEqual(inventory.failed_case_ids, ("Case_1002",))

    def test_filtered_audit_contains_only_valid_cases_in_order(self):
        with self._temporary_directory() as directory:
            root = Path(directory)
            workspace = self._make_workspace(root)
            inventory = build_ood_inventory(workspace)
            output_path = root / "results" / "ood_cases.csv"

            write_ood_case_audit(inventory, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row["case_id"] for row in rows], ["Case_1001", "Case_1003"])
        self.assertEqual(rows[0]["ood_group"], "geometry_ood")

    def test_filtered_audit_accepts_utf8_bom(self):
        with self._temporary_directory() as directory:
            root = Path(directory)
            workspace = self._make_workspace(root)
            original = (workspace / "parameter_audit.csv").read_text(
                encoding="utf-8"
            )
            (workspace / "parameter_audit.csv").write_text(
                original, encoding="utf-8-sig"
            )
            inventory = build_ood_inventory(workspace)

            write_ood_case_audit(inventory, root / "ood_cases.csv")

    def test_inventory_rejects_unaccounted_missing_case(self):
        with self._temporary_directory() as directory:
            workspace = self._make_workspace(Path(directory))
            (workspace / "failed_cases.json").write_text(
                json.dumps({"cases": {}}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "unaccounted"):
                build_ood_inventory(workspace)

    def test_inventory_rejects_valid_failed_overlap(self):
        with self._temporary_directory() as directory:
            workspace = self._make_workspace(Path(directory))
            self._write_hdf5(workspace / "comsol_hdf5" / "Case_1002.h5")

            with self.assertRaisesRegex(ValueError, "both valid and failed"):
                build_ood_inventory(workspace)

    def test_inventory_rejects_invalid_hdf5(self):
        with self._temporary_directory() as directory:
            workspace = self._make_workspace(Path(directory))
            self._write_hdf5(
                workspace / "comsol_hdf5" / "Case_1001.h5",
                include_temperature=False,
            )

            with self.assertRaisesRegex(ValueError, "fields/T"):
                build_ood_inventory(workspace)


class ExplicitEvaluationContextTests(unittest.TestCase):
    def test_context_uses_explicit_ood_data_without_split_manifest(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            data_root = root / "comsol_hdf5"
            data_root.mkdir()
            parameters_json = root / "parameters.json"
            parameters_json.write_text("{}", encoding="utf-8")
            checkpoints = {
                name: root / f"{name}.pt"
                for name in ("meshgraphnet", "transolver")
            }
            for path in checkpoints.values():
                path.write_bytes(b"checkpoint")
            captured = {}

            class FakeDataset:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

            with (
                patch(
                    "evaluation_workspace.common.checkpoint_path",
                    side_effect=lambda name, train_size, seed: checkpoints[name],
                ),
                patch(
                    "evaluation_workspace.common.file_sha256",
                    side_effect=lambda path: f"hash-{Path(path).stem}",
                ),
                patch("evaluation_workspace.common.FpcDataset", FakeDataset),
                patch(
                    "evaluation_workspace.common.choose_device",
                    return_value=torch.device("cpu"),
                ),
                patch(
                    "evaluation_workspace.common.build_graph_transform",
                    return_value="graph-transform",
                ),
            ):
                context = load_evaluation_context_from_cases(
                    models=("meshgraphnet", "transolver"),
                    train_size=100,
                    seed=42,
                    device="auto",
                    data_root=data_root,
                    parameters_json=parameters_json,
                    case_ids=("Case_1001", "Case_1003"),
                    source_name="ood",
                )

        self.assertEqual(captured["data_root"], str(data_root.resolve()))
        self.assertEqual(captured["parameters_json"], str(parameters_json.resolve()))
        self.assertEqual(captured["case_ids"], ["Case_1001", "Case_1003"])
        self.assertEqual(context.manifest["test"], ["Case_1001", "Case_1003"])
        self.assertEqual(context.manifest["source_name"], "ood")
        self.assertEqual(
            context.checkpoint_hashes,
            {
                "meshgraphnet": "hash-meshgraphnet",
                "transolver": "hash-transolver",
            },
        )


class OodTestEntryTests(unittest.TestCase):
    def test_main_isolates_outputs_and_records_133_valid_17_failed(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            workspace = root / "ood_workspace"
            data_root = workspace / "comsol_hdf5"
            parameters_json = workspace / "4_Combined_Master_Dataset.json"
            audit_csv = workspace / "parameter_audit.csv"
            failed_json = workspace / "failed_cases.json"
            output_root = root / "results" / "ood"
            parameter_ids = tuple(f"Case_{index:04d}" for index in range(1001, 1151))
            valid_ids = parameter_ids[:133]
            failed_ids = parameter_ids[133:]
            inventory = OodCaseInventory(
                workspace_root=workspace,
                data_root=data_root,
                parameters_json=parameters_json,
                audit_csv=audit_csv,
                failed_cases_json=failed_json,
                parameter_case_ids=parameter_ids,
                valid_case_ids=valid_ids,
                failed_case_ids=failed_ids,
            )
            context = SimpleNamespace(
                device=torch.device("cpu"),
                checkpoint_hashes={
                    "meshgraphnet": "mgn-hash",
                    "transolver": "trans-hash",
                },
            )
            tables = SimpleNamespace(
                summary=[
                    {"model": "meshgraphnet", "normalized_mse": 1.0, "p_rmse": 2.0, "T_rmse": 3.0},
                    {"model": "transolver", "normalized_mse": 0.5, "p_rmse": 1.0, "T_rmse": 2.0},
                ]
            )

            with (
                patch(
                    "evaluation_workspace.test_ood.build_ood_inventory",
                    return_value=inventory,
                ),
                patch(
                    "evaluation_workspace.test_ood.load_evaluation_context_from_cases",
                    return_value=context,
                ) as load_context,
                patch(
                    "evaluation_workspace.test_ood.materialize_test_predictions"
                ) as materialize,
                patch(
                    "evaluation_workspace.test_ood.analyze_prediction_directory",
                    return_value=tables,
                ) as analyze,
                patch("evaluation_workspace.test_ood.write_evaluation_tables"),
                patch("evaluation_workspace.test_ood.plot_model_comparison"),
                patch(
                    "evaluation_workspace.test_ood.write_ood_case_audit"
                ) as write_audit,
            ):
                ood_test_main(
                    models=("meshgraphnet", "transolver"),
                    train_size=100,
                    seed=42,
                    device="cpu",
                    reuse_predictions=True,
                    overwrite_predictions=False,
                    relative_error_threshold_ratio=0.01,
                    ood_workspace=workspace,
                    output_root=output_root,
                )

            run_dir = output_root / "n0100_seed42"
            config = json.loads(
                (run_dir / "run_config.json").read_text(encoding="utf-8")
            )

        self.assertEqual(config["parameter_case_count"], 150)
        self.assertEqual(config["evaluated_case_count"], 133)
        self.assertEqual(config["failed_case_count"], 17)
        self.assertEqual(config["failed_case_ids"], list(failed_ids))
        self.assertEqual(config["checkpoint_sha256"]["transolver"], "trans-hash")
        self.assertEqual(config["output_directory"], str(run_dir.resolve()))
        self.assertEqual(load_context.call_args.kwargs["case_ids"], list(valid_ids))
        self.assertEqual(materialize.call_args.kwargs["case_ids"], list(valid_ids))
        self.assertEqual(analyze.call_args.kwargs["case_ids"], list(valid_ids))
        self.assertEqual(
            write_audit.call_args.args[1], run_dir / "ood_cases.csv"
        )


if __name__ == "__main__":
    unittest.main()
