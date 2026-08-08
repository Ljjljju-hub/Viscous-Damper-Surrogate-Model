"""Persist COMSOL cases that have already failed during calculation."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).parent.resolve()
LOG_DIR = BASE_DIR / "batch_logs"
REGISTRY_PATH = BASE_DIR / "failed_cases.json"
FAILURE_PATTERN = re.compile(r"\b(Case_\d{4,})\s+计算失败\b")


def load_failure_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.is_file():
        return {"version": 1, "cases": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取失败工况清单 {path}: {error}") from error
    if payload.get("version") != 1 or not isinstance(payload.get("cases"), dict):
        raise RuntimeError(f"失败工况清单格式无效: {path}")
    return payload


def write_failure_registry(payload: dict, path: Path = REGISTRY_PATH) -> None:
    payload["version"] = 1
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def scan_worker_failure_logs(log_dir: Path = LOG_DIR) -> dict[str, str]:
    """Return explicit case failures found in all worker logs."""
    failures: dict[str, str] = {}
    if not log_dir.is_dir():
        return failures
    for log_path in sorted(log_dir.glob("worker_*.log")):
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in FAILURE_PATTERN.finditer(content):
            failures[match.group(1)] = log_path.name
    return failures


def synchronize_failure_registry(
    known_case_ids: Iterable[str],
    completed_case_ids: Iterable[str] = (),
    *,
    log_dir: Path = LOG_DIR,
    registry_path: Path = REGISTRY_PATH,
) -> set[str]:
    """Merge historical logs and remove cases that now have valid output."""
    known = set(known_case_ids)
    completed = set(completed_case_ids)
    payload = load_failure_registry(registry_path)
    cases = payload["cases"]
    before = json.dumps(cases, ensure_ascii=False, sort_keys=True)

    detected_at = time.strftime("%Y-%m-%d %H:%M:%S")
    for case_id, source_log in scan_worker_failure_logs(log_dir).items():
        if case_id in known and case_id not in completed and case_id not in cases:
            cases[case_id] = {
                "detected_at": detected_at,
                "source": source_log,
            }

    for case_id in list(cases):
        if case_id in completed:
            del cases[case_id]

    if json.dumps(cases, ensure_ascii=False, sort_keys=True) != before:
        write_failure_registry(payload, registry_path)
    return set(cases) & known
