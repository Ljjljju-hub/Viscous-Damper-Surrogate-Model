from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pyvista as pv

import failure_registry
import main as worker
import run_remaining as scheduler
from transfer2hdf5 import is_valid_hdf5, vtu_to_hdf5


# Keep temporary paths ASCII-only because some Windows VTK builds cannot write
# through a path component containing Chinese characters.
TEST_TEMP_ROOT = Path(__file__).parents[1].resolve()


class BatchAutomationTest(unittest.TestCase):
    def test_workspace_override_is_forwarded_to_worker_and_converter(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            workspace = Path(directory).resolve()
            model_path = workspace / "source_model.mph"
            try:
                scheduler.configure_workspace(workspace, model_path)
                worker.configure_workspace(workspace, model_path)

                worker_command = scheduler.build_worker_command(
                    ["Case_1001"], cores=4
                )
                converter_command = scheduler.build_converter_command(
                    ["Case_1001"]
                )
                self.assertEqual(
                    worker_command[worker_command.index("--workspace-root") + 1],
                    str(workspace),
                )
                self.assertEqual(
                    worker_command[worker_command.index("--model-path") + 1],
                    str(model_path),
                )
                self.assertEqual(
                    converter_command[converter_command.index("--input-dir") + 1],
                    str(workspace / "comsol_output"),
                )
                self.assertEqual(
                    converter_command[converter_command.index("--output-dir") + 1],
                    str(workspace / "comsol_hdf5"),
                )
                self.assertEqual(worker.PARAMETERS_PATH, workspace / "4_Combined_Master_Dataset.json")
                self.assertEqual(worker.VTU_DIR, workspace / "comsol_output")
            finally:
                scheduler.configure_workspace(None, None)
                worker.configure_workspace(None, None)

    def test_default_workspace_remains_the_calculation_directory(self):
        scheduler.configure_workspace(None, None)
        worker.configure_workspace(None, None)

        self.assertEqual(scheduler.WORKSPACE_ROOT, scheduler.SCRIPT_DIR)
        self.assertEqual(worker.WORKSPACE_ROOT, worker.SCRIPT_DIR)
        self.assertEqual(
            scheduler.PARAMETERS_PATH,
            scheduler.SCRIPT_DIR / "4_Combined_Master_Dataset.json",
        )
        self.assertEqual(worker.MODEL_PATH, worker.SCRIPT_DIR / "standard_model.mph")

    def test_load_select_and_chunk_cases(self):
        samples = [
            {"case_id": f"Case_{index:04d}"} for index in range(1, 7)
        ]
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            path = Path(directory) / "parameters.json"
            path.write_text(
                json.dumps({"parameters_list": samples}), encoding="utf-8"
            )
            case_ids = scheduler.load_case_ids(path)

        self.assertEqual(case_ids[0], "Case_0001")
        self.assertEqual(
            scheduler.select_case_range(case_ids, 2, 5),
            ["Case_0002", "Case_0003", "Case_0004", "Case_0005"],
        )
        self.assertEqual(
            list(scheduler.chunked(case_ids, 4)),
            [case_ids[:4], case_ids[4:]],
        )
        selected = worker.select_samples(
            samples, case_ids=["Case_0005", "Case_0002"]
        )
        self.assertEqual(
            [sample["case_id"] for sample in selected],
            ["Case_0005", "Case_0002"],
        )

    def test_inject_parameters_uses_comsol_units(self):
        class FakeModel:
            def __init__(self):
                self.values = {}

            def parameter(self, name, value):
                self.values[name] = value

        model = FakeModel()
        worker.inject_parameters(
            model,
            {
                "geometry": {
                    "c": 1,
                    "sx": 2,
                    "sy": 3,
                    "r1": 4,
                    "a2": 5,
                    "b1": 6,
                    "b2": 7,
                },
                "loading": {"A": 8, "Ts": 9},
                "material": {"mu": 10},
            },
        )
        self.assertEqual(model.values["A"], "8 [mm]")
        self.assertEqual(model.values["Ts"], "9 [s]")
        self.assertEqual(model.values["mu_0"], "10 [Pa*s]")

    def test_shutdown_disconnects_client_and_stops_server(self):
        client = mock.Mock()
        server = mock.Mock()
        server.running.return_value = True
        with mock.patch.object(worker.mph_session, "server", server):
            worker.shutdown_comsol(client)
        client.disconnect.assert_called_once_with()
        server.stop.assert_called_once_with(timeout=30)

    def test_worker_command_reuses_current_pinn_python(self):
        command = scheduler.build_worker_command(["Case_0001"], cores=8)
        self.assertEqual(command[0], scheduler.sys.executable)
        self.assertEqual(command[-2:], ["--cores", "8"])
        self.assertIn("Case_0001", command)

    def test_vtu_conversion_is_structurally_valid(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            vtu_path = root / "Case_0001.vtu"
            h5_path = root / "Case_0001.h5"
            points = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            )
            grid = pv.UnstructuredGrid(
                {pv.CellType.TRIANGLE: np.array([[0, 1, 2]])}, points
            )
            for time_value in (0.0, 0.1):
                grid.point_data[f"p_@_t={time_value}"] = np.full(3, time_value + 1)
                grid.point_data[f"T_@_t={time_value}"] = np.full(3, time_value + 2)
            grid.save(vtu_path)

            fields = vtu_to_hdf5(vtu_path, h5_path)

            self.assertEqual(fields, ["T", "p"])
            self.assertTrue(is_valid_hdf5(h5_path))
            with h5py.File(h5_path, "r") as handle:
                self.assertEqual(handle["fields/p"].shape, (2, 3))
            self.assertFalse(list(root.glob("*.partial.h5")))

    def test_pending_detection_accepts_valid_hdf5_or_vtu(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            h5_dir = root / "h5"
            vtu_dir = root / "vtu"
            h5_dir.mkdir()
            vtu_dir.mkdir()
            valid_vtu = vtu_dir / "Case_0002.vtu"
            valid_vtu.write_bytes(
                b"<VTKFile>" + b"x" * 2048 + b"</VTKFile>"
            )
            with mock.patch.object(scheduler, "HDF5_DIR", h5_dir), mock.patch.object(
                scheduler, "VTU_DIR", vtu_dir
            ):
                self.assertFalse(scheduler.hdf5_complete("Case_0001"))
                self.assertTrue(scheduler.vtu_complete("Case_0002"))
                self.assertEqual(
                    scheduler.unresolved_cases(
                        ["Case_0001", "Case_0002"], convert=False
                    ),
                    ["Case_0001"],
                )

    def test_failed_cases_are_skipped_unless_explicitly_retried(self):
        case_ids = ["Case_0001", "Case_0002", "Case_0003"]
        pending = scheduler.compute_pending_cases(
            case_ids,
            hdf5_done={"Case_0001"},
            vtu_only=set(),
            failed_cases={"Case_0002"},
        )
        self.assertEqual(pending, ["Case_0003"])
        self.assertEqual(
            scheduler.compute_pending_cases(
                case_ids,
                hdf5_done={"Case_0001"},
                vtu_only=set(),
                failed_cases={"Case_0002"},
                retry_failed=True,
            ),
            ["Case_0002", "Case_0003"],
        )

    def test_failure_registry_recovers_logs_and_prunes_completed_cases(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            log_dir = root / "logs"
            log_dir.mkdir()
            registry_path = root / "failed_cases.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "cases": {"Case_9999": {"source": "earlier_range"}},
                    }
                ),
                encoding="utf-8",
            )
            (log_dir / "worker_test.log").write_text(
                "2026-08-08 | ERROR | Case_0001 计算失败\n"
                "2026-08-08 | ERROR | Case_0002 计算失败\n",
                encoding="utf-8",
            )

            failed = failure_registry.synchronize_failure_registry(
                ["Case_0001", "Case_0002", "Case_0003"],
                completed_case_ids={"Case_0002"},
                log_dir=log_dir,
                registry_path=registry_path,
            )

            self.assertEqual(failed, {"Case_0001"})
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["cases"]), {"Case_0001", "Case_9999"})

    def test_worker_does_not_skip_large_but_corrupt_output(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            h5_dir = root / "h5"
            vtu_dir = root / "vtu"
            h5_dir.mkdir()
            vtu_dir.mkdir()
            (h5_dir / "Case_0001.h5").write_bytes(b"broken" * 1024)
            with mock.patch.object(worker, "HDF5_DIR", h5_dir), mock.patch.object(
                worker, "VTU_DIR", vtu_dir
            ):
                self.assertFalse(worker.output_exists("Case_0001"))

                (vtu_dir / "Case_0001.vtu").write_bytes(
                    b"<VTKFile>" + b"x" * 2048 + b"</VTKFile>"
                )
                self.assertTrue(worker.output_exists("Case_0001"))


if __name__ == "__main__":
    unittest.main()
