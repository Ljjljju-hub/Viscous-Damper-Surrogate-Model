from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv

from .metrics import case_relative_threshold, relative_error_mask
from .visualization_pipeline import FIELD_NAMES, TemporalComparison


def export_comparison_pvd(
    sequence: TemporalComparison,
    output_dir: Path,
    threshold_ratio: float,
) -> Path:
    sequence.validate()
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [
        case_relative_threshold(sequence.truth[..., index], threshold_ratio)
        for index in range(len(FIELD_NAMES))
    ]
    root = ET.Element(
        "VTKFile",
        {"type": "Collection", "version": "0.1", "byte_order": "LittleEndian"},
    )
    collection = ET.SubElement(root, "Collection")
    for frame_offset, physical_time in enumerate(sequence.time_steps):
        positions = sequence.positions[frame_offset]
        points = np.column_stack([positions, np.zeros(len(positions))])
        grid = pv.UnstructuredGrid(
            {pv.CellType.TRIANGLE: sequence.face.T}, points
        )
        grid.point_data["mesh_region"] = sequence.region.astype(np.int32)
        grid.point_data["mesh_velocity"] = np.column_stack(
            [sequence.velocity[frame_offset], np.zeros(len(positions))]
        )
        for field_index, field_name in enumerate(FIELD_NAMES):
            truth = sequence.truth[frame_offset, :, field_index]
            grid.point_data[f"{field_name}_ground_truth"] = truth
            valid = relative_error_mask(truth, thresholds[field_index])
            grid.point_data[f"relative_error_valid_{field_name}"] = valid.astype(np.uint8)
            for model_name, prediction_values in sequence.predictions.items():
                prediction = prediction_values[frame_offset, :, field_index]
                absolute = np.abs(prediction - truth)
                relative = np.full(truth.shape, np.nan, dtype=np.float32)
                relative[valid] = absolute[valid] / np.abs(truth[valid]) * 100.0
                grid.point_data[f"{field_name}_{model_name}"] = prediction
                grid.point_data[f"{field_name}_abs_error_{model_name}"] = absolute
                grid.point_data[f"{field_name}_relative_error_{model_name}"] = relative
        filename = f"frame_{frame_offset:04d}.vtu"
        grid.save(frame_dir / filename)
        ET.SubElement(
            collection,
            "DataSet",
            {
                "timestep": f"{float(physical_time):.17g}",
                "group": "",
                "part": "0",
                "file": f"frames/{filename}",
            },
        )
    pvd_path = output_dir / "comparison.pvd"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(pvd_path, encoding="utf-8", xml_declaration=True)
    return pvd_path
