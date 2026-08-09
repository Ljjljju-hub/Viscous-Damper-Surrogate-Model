from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

from .common import EvaluationContext
from .plotting import plot_error_vs_time
from .visualization_pipeline import (
    build_step_metric_rows,
    load_saved_one_step_sequence,
    write_step_metrics,
)
from .vtu_export import export_comparison_pvd


FIELD_NAMES = ("p", "T")
SELECTION_METRICS = (
    "rmse",
    "max_absolute_error",
    "max_relative_error_percent",
)


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select_representative_cases(
    case_metrics_path: Path, models: list[str] | tuple[str, ...]
) -> list[dict]:
    source_rows = _read_csv(case_metrics_path)
    selections = []
    for model_name in models:
        model_rows = [row for row in source_rows if row["model"] == model_name]
        if not model_rows:
            raise ValueError(f"No case metrics found for model {model_name!r}.")
        for field_name in FIELD_NAMES:
            for metric_name in SELECTION_METRICS:
                column = f"{field_name}_{metric_name}"
                try:
                    ranked = sorted(
                        model_rows,
                        key=lambda row: (float(row[column]), row["case_id"]),
                    )
                except KeyError as error:
                    raise ValueError(
                        f"case_metrics.csv is missing column {column!r}."
                    ) from error
                for extreme, source in (("min", ranked[0]), ("max", ranked[-1])):
                    selections.append(
                        {
                            "model": model_name,
                            "field": field_name,
                            "metric": metric_name,
                            "extreme": extreme,
                            "case_id": source["case_id"],
                            "value": float(source[column]),
                            "source_column": column,
                            "definition": (
                                "minimum across each case's worst valid point"
                                if extreme == "min"
                                and metric_name.startswith("max_")
                                else f"case-level {metric_name} {extreme}"
                            ),
                        }
                    )
    return selections


def export_representative_cases(
    context: EvaluationContext,
    selections: list[dict],
    prediction_root: Path,
    output_root: Path,
    threshold_ratio: float,
) -> list[Path]:
    output_root = Path(output_root)
    _write_csv_atomic(output_root / "representative_cases.csv", selections)
    case_ids = list(dict.fromkeys(row["case_id"] for row in selections))
    exported = []
    for case_number, case_id in enumerate(case_ids, 1):
        print(f"EXPORT representative {case_id} ({case_number}/{len(case_ids)})")
        case_dir = output_root / "representative_cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        reasons = [row for row in selections if row["case_id"] == case_id]
        _write_csv_atomic(case_dir / "selection_reasons.csv", reasons)

        ground_truth_dir = case_dir / "ground_truth"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        source_h5 = context.dataset._resolve_file(case_id)
        shutil.copy2(source_h5, ground_truth_dir / source_h5.name)

        sequence = load_saved_one_step_sequence(
            context,
            prediction_root,
            case_id,
            start_index=0,
            steps=None,
        )
        step_rows = build_step_metric_rows(sequence, threshold_ratio)
        write_step_metrics(step_rows, case_dir / "step_metrics.csv")
        plot_error_vs_time(step_rows, case_dir / "error_vs_time.png")
        exported.append(
            export_comparison_pvd(sequence, case_dir, threshold_ratio)
        )
    return exported
