import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        default=PROJECT_ROOT
        / "experiments"
        / "dataset_scale"
        / "split_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--expected-cases", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    parameters_json = args.parameters_json.resolve()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; use --force to replace it.")

    parameters = load_case_parameters(parameters_json)
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

    case_count = len(snapshots)
    if case_count != args.expected_cases and not args.allow_incomplete:
        raise RuntimeError(
            f"Expected {args.expected_cases} valid cases, found {case_count}. "
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

    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "temporary_incomplete": case_count != args.expected_cases,
        "seed": args.seed,
        "expected_cases": args.expected_cases,
        "case_count": case_count,
        "data_root": str(data_root),
        "parameters_json": str(parameters_json),
        "parameters_json_sha256": sha256_file(parameters_json),
        "train_pool": train_pool,
        "valid": valid,
        "test": test,
        "invalid_cases": invalid,
        "snapshot": snapshots,
    }
    atomic_write_json(output, manifest)
    print(f"saved={output}")
    print(
        f"valid_cases={case_count} train_pool={len(train_pool)} "
        f"valid={len(valid)} test={len(test)} invalid={len(invalid)}"
    )


if __name__ == "__main__":
    main()
