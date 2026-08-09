from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass(frozen=True)
class OodCaseInventory:
    workspace_root: Path
    data_root: Path
    parameters_json: Path
    audit_csv: Path
    failed_cases_json: Path
    parameter_case_ids: tuple[str, ...]
    valid_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required OOD file does not exist: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read OOD JSON: {path}") from error


def _validate_unique_case_ids(values: list[str], source: str) -> tuple[str, ...]:
    if not values or any(not case_id for case_id in values):
        raise ValueError(f"{source} must contain non-empty case IDs.")
    if len(values) != len(set(values)):
        raise ValueError(f"{source} contains duplicate case IDs.")
    return tuple(values)


def _validate_hdf5(path: Path) -> None:
    required = (
        "time_steps",
        "mesh/coordinates",
        "mesh/connectivity",
        "fields/p",
        "fields/T",
    )
    try:
        with h5py.File(path, "r") as handle:
            for name in required:
                if name not in handle:
                    raise ValueError(f"{path.name} is missing required dataset {name}.")
            times = handle["time_steps"]
            coordinates = handle["mesh/coordinates"]
            connectivity = handle["mesh/connectivity"]
            pressure = handle["fields/p"]
            temperature = handle["fields/T"]
            if times.ndim != 1 or len(times) < 2:
                raise ValueError(f"{path.name} must contain at least two time_steps.")
            if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                raise ValueError(f"{path.name} mesh/coordinates must have shape [N, >=2].")
            if connectivity.size == 0:
                raise ValueError(f"{path.name} mesh/connectivity must not be empty.")
            expected_shape = (len(times), coordinates.shape[0])
            if pressure.shape != expected_shape:
                raise ValueError(
                    f"{path.name} fields/p shape {pressure.shape} != {expected_shape}."
                )
            if temperature.shape != expected_shape:
                raise ValueError(
                    f"{path.name} fields/T shape {temperature.shape} != {expected_shape}."
                )
    except OSError as error:
        raise ValueError(f"Cannot open OOD HDF5: {path}") from error


def build_ood_inventory(workspace_root: Path) -> OodCaseInventory:
    workspace = Path(workspace_root).resolve()
    data_root = workspace / "comsol_hdf5"
    parameters_json = workspace / "4_Combined_Master_Dataset.json"
    audit_csv = workspace / "parameter_audit.csv"
    failed_cases_json = workspace / "failed_cases.json"

    parameters_payload = _load_json(parameters_json)
    parameter_rows = parameters_payload.get("parameters_list")
    if not isinstance(parameter_rows, list):
        raise ValueError("OOD parameters_list must be a list.")
    parameter_case_ids = _validate_unique_case_ids(
        [str(row.get("case_id", "")) for row in parameter_rows],
        "OOD parameters_list",
    )
    parameter_set = set(parameter_case_ids)

    failed_payload = _load_json(failed_cases_json)
    failed_mapping = failed_payload.get("cases")
    if not isinstance(failed_mapping, dict):
        raise ValueError("failed_cases.json cases must be an object.")
    failed_case_ids = _validate_unique_case_ids(
        sorted(str(case_id) for case_id in failed_mapping),
        "failed_cases.json",
    ) if failed_mapping else ()
    failed_set = set(failed_case_ids)

    if not data_root.is_dir():
        raise FileNotFoundError(f"OOD HDF5 directory does not exist: {data_root}")
    hdf5_paths = sorted(data_root.glob("Case_*.h5"))
    valid_case_ids = _validate_unique_case_ids(
        [path.stem for path in hdf5_paths],
        "OOD HDF5 directory",
    )
    valid_set = set(valid_case_ids)

    unknown_valid = sorted(valid_set - parameter_set)
    unknown_failed = sorted(failed_set - parameter_set)
    if unknown_valid or unknown_failed:
        raise ValueError(
            f"OOD results reference unknown parameters: "
            f"valid={unknown_valid}, failed={unknown_failed}."
        )
    overlap = sorted(valid_set & failed_set)
    if overlap:
        raise ValueError(f"OOD cases are both valid and failed: {overlap}.")
    unaccounted = sorted(parameter_set - valid_set - failed_set)
    if unaccounted:
        raise ValueError(f"OOD parameter cases are unaccounted: {unaccounted}.")

    for path in hdf5_paths:
        _validate_hdf5(path)
    if not audit_csv.is_file():
        raise FileNotFoundError(f"OOD audit CSV does not exist: {audit_csv}")

    return OodCaseInventory(
        workspace_root=workspace,
        data_root=data_root,
        parameters_json=parameters_json,
        audit_csv=audit_csv,
        failed_cases_json=failed_cases_json,
        parameter_case_ids=parameter_case_ids,
        valid_case_ids=valid_case_ids,
        failed_case_ids=failed_case_ids,
    )


def write_ood_case_audit(inventory: OodCaseInventory, output_path: Path) -> None:
    with inventory.audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "case_id" not in reader.fieldnames:
            raise ValueError("OOD audit CSV must contain a case_id column.")
        rows_by_case = {}
        for row in reader:
            case_id = row.get("case_id", "")
            if case_id in rows_by_case:
                raise ValueError(f"OOD audit CSV contains duplicate {case_id}.")
            rows_by_case[case_id] = row
        fieldnames = list(reader.fieldnames)

    missing = [
        case_id for case_id in inventory.valid_case_ids if case_id not in rows_by_case
    ]
    if missing:
        raise ValueError(f"OOD audit CSV is missing valid cases: {missing}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                rows_by_case[case_id] for case_id in inventory.valid_case_ids
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
