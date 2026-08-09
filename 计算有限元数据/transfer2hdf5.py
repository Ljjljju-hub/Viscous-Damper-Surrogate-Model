"""Incrementally convert COMSOL VTU exports to atomic HDF5 files."""

from __future__ import annotations

import argparse
import os
import re
import traceback
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_INPUT_DIR = BASE_DIR / "comsol_output"
DEFAULT_OUTPUT_DIR = BASE_DIR / "comsol_hdf5"
FIELD_PATTERN = re.compile(r"(.+)_@_t=(.*)")


def is_valid_hdf5(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    try:
        with h5py.File(path, "r") as handle:
            coordinates = handle["mesh/coordinates"]
            connectivity = handle["mesh/connectivity"]
            time_steps = handle["time_steps"]
            fields = handle["fields"]
            if coordinates.ndim != 2 or coordinates.shape[0] == 0:
                return False
            if connectivity.size == 0 or time_steps.ndim != 1 or time_steps.size == 0:
                return False
            if not fields.keys():
                return False
            return all(
                dataset.shape == (time_steps.size, coordinates.shape[0])
                for dataset in fields.values()
            )
    except (OSError, KeyError, ValueError):
        return False


def vtu_to_hdf5(vtu_filepath: Path, h5_filepath: Path) -> list[str]:
    """Convert one VTU and atomically publish the resulting HDF5 file."""
    print(f"读取 {vtu_filepath.name}")
    mesh = pv.read(str(vtu_filepath))
    fields_by_time: dict[str, dict[float, np.ndarray]] = {}
    time_steps: set[float] = set()

    for name in mesh.point_data.keys():
        match = FIELD_PATTERN.fullmatch(name)
        if not match:
            continue
        field_name = match.group(1).strip()
        time_value = float(match.group(2))
        values = np.asarray(mesh.point_data[name]).squeeze()
        if values.shape != (mesh.number_of_points,):
            raise ValueError(
                f"场 {name} 不是节点标量场，形状为 {values.shape}"
            )
        time_steps.add(time_value)
        fields_by_time.setdefault(field_name, {})[time_value] = values

    if not time_steps or not fields_by_time:
        raise ValueError(f"{vtu_filepath.name} 中没有解析到时间步节点场")

    sorted_times = sorted(time_steps)
    for field_name, values_by_time in fields_by_time.items():
        missing_times = [time_value for time_value in sorted_times if time_value not in values_by_time]
        if missing_times:
            raise ValueError(
                f"场 {field_name} 缺少 {len(missing_times)} 个时间步，拒绝用零值填充"
            )

    h5_filepath.parent.mkdir(parents=True, exist_ok=True)
    partial = h5_filepath.with_name(
        f".{h5_filepath.stem}.{os.getpid()}.partial.h5"
    )
    partial.unlink(missing_ok=True)
    try:
        with h5py.File(partial, "w") as handle:
            mesh_group = handle.create_group("mesh")
            mesh_group.create_dataset(
                "coordinates", data=mesh.points, compression="gzip"
            )
            mesh_group.create_dataset(
                "connectivity", data=mesh.cells, compression="gzip"
            )
            handle.create_dataset(
                "time_steps", data=np.asarray(sorted_times), compression="gzip"
            )
            fields_group = handle.create_group("fields")
            for field_name, values_by_time in fields_by_time.items():
                matrix = np.stack(
                    [values_by_time[time_value] for time_value in sorted_times]
                ).astype(np.float32, copy=False)
                fields_group.create_dataset(
                    field_name, data=matrix, compression="gzip"
                )
            handle.flush()

        if not is_valid_hdf5(partial):
            raise RuntimeError(f"生成后的 HDF5 结构校验失败: {partial}")
        os.replace(partial, h5_filepath)
    finally:
        partial.unlink(missing_ok=True)

    extracted = sorted(fields_by_time)
    print(
        f"完成 {h5_filepath.name}: {mesh.number_of_points} 节点, "
        f"{len(sorted_times)} 步, fields={extracted}"
    )
    return extracted


def batch_convert_dir(
    input_dir: Path,
    output_dir: Path,
    case_ids: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"VTU 输入目录不存在: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if case_ids:
        vtu_files = [input_dir / f"{case_id}.vtu" for case_id in case_ids]
        vtu_files = [path for path in vtu_files if path.is_file()]
    else:
        vtu_files = sorted(input_dir.glob("Case_*.vtu"))

    converted = 0
    skipped = 0
    failed = 0
    for index, vtu_file in enumerate(vtu_files, 1):
        h5_file = output_dir / f"{vtu_file.stem}.h5"
        if not overwrite and is_valid_hdf5(h5_file):
            skipped += 1
            continue
        print(f"[{index}/{len(vtu_files)}] {vtu_file.name}")
        try:
            vtu_to_hdf5(vtu_file, h5_file)
            converted += 1
        except Exception:
            failed += 1
            print(f"转换失败: {vtu_file.name}\n{traceback.format_exc()}")

    print(f"转换汇总: 新增={converted}, 跳过={skipped}, 失败={failed}")
    return converted, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量转换 COMSOL VTU 为 HDF5")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ids", help="只转换逗号分隔的 case_id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_ids = None
    if args.case_ids:
        case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    try:
        _, _, failed = batch_convert_dir(
            args.input_dir, args.output_dir, case_ids, args.overwrite
        )
    except Exception:
        traceback.print_exc()
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
