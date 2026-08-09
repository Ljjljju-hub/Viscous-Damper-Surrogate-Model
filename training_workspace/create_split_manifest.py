import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.dataset import load_case_parameters
from meshGraphNet_self.experiment_utils import atomic_write_json


REQUIRED_DATASETS = (
    "mesh/coordinates",
    "mesh/connectivity",
    "time_steps",
    "fields/p",
    "fields/T",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_case_file(path: Path) -> dict:
    with h5py.File(path, "r") as h5_file:
        missing = [name for name in REQUIRED_DATASETS if name not in h5_file]
        if missing:
            raise ValueError(f"missing datasets: {missing}")
        node_count = int(h5_file["mesh/coordinates"].shape[0])
        time_count = int(h5_file["time_steps"].shape[0])
        if time_count < 2:
            raise ValueError("time_steps contains fewer than two frames")
        for field_name in ("p", "T"):
            shape = tuple(h5_file[f"fields/{field_name}"].shape)
            if shape != (time_count, node_count):
                raise ValueError(
                    f"fields/{field_name} shape={shape}, "
                    f"expected {(time_count, node_count)}"
                )
    stat = path.stat()
    return {
        "case_id": path.stem,
        "file": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "nodes": node_count,
        "time_steps": time_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze valid HDF5 cases and create fixed train/valid/test splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "计算有限元数据" / "comsol_hdf5",
    )
    parser.add_argument(
        "--parameters-json",
        type=Path,
        default=PROJECT_ROOT / "计算有限元数据" / "4_Combined_Master_Dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "dataset_split" / "split_manifest.json",
    )
    parser.add_argument(
        "--portable-output",
        type=Path,
        default=WORKSPACE_ROOT / "dataset_split" / "case_split.json",
    )
    parser.add_argument(
        "--case-index-output",
        type=Path,
        default=WORKSPACE_ROOT / "dataset_split" / "case_index.csv",
    )
    parser.add_argument(
        "--failed-registry",
        type=Path,
        default=PROJECT_ROOT / "计算有限元数据" / "failed_cases.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-count", type=int, default=80)
    parser.add_argument("--test-count", type=int, default=81)
    parser.add_argument("--expected-total-cases", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_failed_case_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"Invalid failed-case registry: {path}")
    return set(cases)


def write_case_index(
    path: Path,
    parameter_case_ids: list[str],
    snapshots: list[dict],
    train_pool: list[str],
    valid: list[str],
    test: list[str],
    failed: list[str],
) -> None:
    snapshot_by_id = {item["case_id"]: item for item in snapshots}
    train_order = {case_id: index for index, case_id in enumerate(train_pool, 1)}
    valid_set = set(valid)
    test_set = set(test)
    failed_set = set(failed)
    rows = []
    for case_id in parameter_case_ids:
        snapshot = snapshot_by_id.get(case_id)
        order = train_order.get(case_id)
        if order is not None:
            status = "valid_hdf5"
            split = "train"
        elif case_id in valid_set:
            status = "valid_hdf5"
            split = "valid"
        elif case_id in test_set:
            status = "valid_hdf5"
            split = "test"
        elif case_id in failed_set:
            status = "failed_terminal"
            split = "excluded"
        else:
            status = "unresolved"
            split = "excluded"
        rows.append(
            {
                "case_id": case_id,
                "status": status,
                "split": split,
                "train_order": order or "",
                "hdf5_file": snapshot["file"] if snapshot else "",
                "nodes": snapshot["nodes"] if snapshot else "",
                "time_steps": snapshot["time_steps"] if snapshot else "",
                "size_bytes": snapshot["size"] if snapshot else "",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    parameters_json = args.parameters_json.resolve()
    output = args.output.resolve()
    portable_output = args.portable_output.resolve()
    case_index_output = args.case_index_output.resolve()
    failed_registry = args.failed_registry.resolve()
    existing_outputs = [
        path
        for path in (output, portable_output, case_index_output)
        if path.exists()
    ]
    if existing_outputs and not args.force:
        raise FileExistsError(
            f"Outputs already exist: {existing_outputs}; use --force to replace them."
        )

    parameters = load_case_parameters(parameters_json)
    parameter_case_ids = list(parameters)
    if len(parameter_case_ids) != args.expected_total_cases:
        raise RuntimeError(
            f"Expected {args.expected_total_cases} parameter cases, "
            f"found {len(parameter_case_ids)}."
        )
    files = sorted(data_root.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {data_root}.")

    snapshots = []
    invalid = []
    for path in files:
        if path.stem not in parameters:
            invalid.append({"case_id": path.stem, "reason": "missing JSON parameters"})
            continue
        try:
            snapshots.append(validate_case_file(path))
        except (OSError, ValueError) as exc:
            invalid.append({"case_id": path.stem, "reason": str(exc)})

    valid_case_ids = {item["case_id"] for item in snapshots}
    registered_failed = load_failed_case_ids(failed_registry)
    failed = [
        case_id
        for case_id in parameter_case_ids
        if case_id in registered_failed and case_id not in valid_case_ids
    ]
    unresolved = [
        case_id
        for case_id in parameter_case_ids
        if case_id not in valid_case_ids and case_id not in registered_failed
    ]
    case_count = len(snapshots)
    if unresolved and not args.allow_incomplete:
        raise RuntimeError(
            f"Found {len(unresolved)} unresolved cases: {unresolved[:20]}. "
            "Wait for COMSOL or use --allow-incomplete for a temporary manifest."
        )
    if args.valid_count + args.test_count >= case_count:
        raise ValueError("valid-count + test-count must be smaller than case count.")

    case_ids = [item["case_id"] for item in snapshots]
    random.Random(args.seed).shuffle(case_ids)
    valid_end = args.valid_count
    test_end = valid_end + args.test_count
    valid = case_ids[:valid_end]
    test = case_ids[valid_end:test_end]
    train_pool = case_ids[test_end:]
    train_sizes = list(range(100, len(train_pool) + 1, 100))

    manifest = {
        "version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "temporary_incomplete": bool(unresolved),
        "seed": args.seed,
        "expected_total_cases": args.expected_total_cases,
        "parameter_case_count": len(parameter_case_ids),
        "case_count": case_count,
        "failed_terminal_count": len(failed),
        "failed_terminal": failed,
        "unresolved_cases": unresolved,
        "data_root": str(data_root),
        "parameters_json": str(parameters_json),
        "parameters_json_sha256": sha256_file(parameters_json),
        "train_pool": train_pool,
        "valid": valid,
        "test": test,
        "train_sizes": train_sizes,
        "invalid_cases": invalid,
        "snapshot": snapshots,
    }
    atomic_write_json(output, manifest)
    portable_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"data_root", "parameters_json", "snapshot"}
    }
    portable_manifest["data_root"] = "计算有限元数据/comsol_hdf5"
    portable_manifest["parameters_json"] = (
        "计算有限元数据/4_Combined_Master_Dataset.json"
    )
    portable_manifest["runtime_manifest"] = (
        "training_workspace/dataset_split/split_manifest.json"
    )
    atomic_write_json(portable_output, portable_manifest)
    write_case_index(
        case_index_output,
        parameter_case_ids,
        snapshots,
        train_pool,
        valid,
        test,
        failed,
    )
    print(f"saved={output}")
    print(f"portable={portable_output}")
    print(f"case_index={case_index_output}")
    print(
        f"valid_cases={case_count} train_pool={len(train_pool)} "
        f"valid={len(valid)} test={len(test)} failed={len(failed)} "
        f"unresolved={len(unresolved)} invalid_files={len(invalid)}"
    )


if __name__ == "__main__":
    main()
