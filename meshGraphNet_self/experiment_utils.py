import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_split_manifest(path: Path) -> dict:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("train_pool", "valid", "test")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Split manifest is missing keys: {missing}")

    groups = {key: list(manifest[key]) for key in required}
    for key, values in groups.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Split manifest group {key!r} contains duplicates.")
    if set(groups["train_pool"]) & set(groups["valid"]):
        raise ValueError("train_pool and valid overlap.")
    if set(groups["train_pool"]) & set(groups["test"]):
        raise ValueError("train_pool and test overlap.")
    if set(groups["valid"]) & set(groups["test"]):
        raise ValueError("valid and test overlap.")
    manifest["_path"] = str(manifest_path)
    return manifest


def verify_manifest_snapshot(manifest: dict) -> list:
    """Return human-readable errors when frozen source files have changed."""
    errors = []
    data_root = Path(manifest["data_root"])
    snapshots = {item["case_id"]: item for item in manifest.get("snapshot", [])}
    expected_ids = [
        *manifest["train_pool"],
        *manifest["valid"],
        *manifest["test"],
    ]
    for case_id in expected_ids:
        expected = snapshots.get(case_id)
        if expected is None:
            errors.append(f"{case_id}: missing snapshot metadata")
            continue
        path = data_root / expected["file"]
        if not path.exists():
            errors.append(f"{case_id}: HDF5 file is missing")
            continue
        stat = path.stat()
        if stat.st_size != int(expected["size"]):
            errors.append(f"{case_id}: file size changed")
        if stat.st_mtime_ns != int(expected["mtime_ns"]):
            errors.append(f"{case_id}: modification time changed")

    parameters_path = Path(manifest["parameters_json"])
    if not parameters_path.exists():
        errors.append("parameters JSON is missing")
    elif manifest.get("parameters_json_sha256"):
        digest = hashlib.sha256(parameters_path.read_bytes()).hexdigest()
        if digest != manifest["parameters_json_sha256"]:
            errors.append("parameters JSON SHA256 changed")
    return errors


def select_manifest_cases(
    manifest_path: Path, train_size: Optional[int]
) -> Tuple[list, list, list, dict]:
    manifest = load_split_manifest(manifest_path)
    train_pool = list(manifest["train_pool"])
    selected_size = len(train_pool) if train_size is None else int(train_size)
    if selected_size < 1 or selected_size > len(train_pool):
        raise ValueError(
            f"train_size must be in [1, {len(train_pool)}], got {selected_size}."
        )
    return (
        train_pool[:selected_size],
        list(manifest["valid"]),
        list(manifest["test"]),
        manifest,
    )


def capture_rng_state(
    data_loader_generator: Optional[torch.Generator] = None,
) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    if data_loader_generator is not None:
        state["data_loader_generator"] = data_loader_generator.get_state()
    return state


def restore_rng_state(
    state: Optional[dict],
    data_loader_generator: Optional[torch.Generator] = None,
) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["torch_cuda"]]
        )
    if data_loader_generator is not None and "data_loader_generator" in state:
        data_loader_generator.set_state(state["data_loader_generator"].cpu())


def upsert_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: Dict[int, Dict[str, str]] = {}
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
            for existing in reader:
                rows[int(existing["epoch"])] = existing
    rows[int(row["epoch"])] = {key: str(value) for key, value in row.items()}

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in sorted(rows):
            writer.writerow(rows[epoch])
    os.replace(temporary, path)


def read_metrics_rows(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def case_ids_from_files(files: Iterable[Path]) -> list:
    return [Path(path).stem for path in files]
