import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv

from evaluation_workspace.visualization_pipeline import TemporalComparison
from evaluation_workspace.vtu_export import export_comparison_pvd


class VtuExportTests(unittest.TestCase):
    def test_dynamic_points_fields_and_pvd_times(self):
        truth = np.array(
            [
                [[1.0, 300.0], [2.0, 301.0], [3.0, 302.0]],
                [[2.0, 301.0], [3.0, 302.0], [4.0, 303.0]],
            ],
            dtype=np.float32,
        )
        positions = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.1], [1.0, 0.1], [0.0, 1.1]],
            ],
            dtype=np.float32,
        )
        sequence = TemporalComparison(
            case_id="Case_0001",
            source_mode="saved_one_step",
            time_indices=np.array([0, 1]),
            time_steps=np.array([0.0, 0.1]),
            positions=positions,
            velocity=np.zeros((2, 3, 2), dtype=np.float32),
            face=np.array([[0], [1], [2]], dtype=np.int64),
            region=np.array([0, 1, 2]),
            truth=truth,
            predictions={"meshgraphnet": truth + 1.0},
            output_std={"meshgraphnet": np.array([2.0, 1.0])},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pvd = export_comparison_pvd(sequence, output, 0.01)
            frame = pv.read(output / "frames" / "frame_0001.vtu")
            tree = ET.parse(pvd)

        np.testing.assert_allclose(frame.points[:, :2], positions[1])
        self.assertIn("p_ground_truth", frame.point_data)
        self.assertIn("p_meshgraphnet", frame.point_data)
        datasets = tree.findall("./Collection/DataSet")
        self.assertEqual([item.attrib["timestep"] for item in datasets], ["0", "0.10000000000000001"])
        self.assertEqual(datasets[1].attrib["file"], "frames/frame_0001.vtu")


if __name__ == "__main__":
    unittest.main()
