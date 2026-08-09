"""Generate and validate grouped out-of-domain damper parameters."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
from scipy.stats import qmc


TRAINING_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "geometry": {
        "c": (1.0, 3.0),
        "sx": (40.0, 120.0),
        "sy": (120.0, 320.0),
        "r1": (50.0, 70.0),
        "a2": (40.0, 80.0),
        "b1": (80.0, 120.0),
        "b2": (80.0, 160.0),
    },
    "loading": {
        "A": (10.0, 90.0),
        "Ts": (0.1, 0.5),
    },
    "material": {"mu": (1000.0, 3000.0)},
}
PARAMETER_GROUPS = {
    group: tuple(bounds) for group, bounds in TRAINING_BOUNDS.items()
}
OOD_GROUPS = {
    "geometry_ood": "geometry",
    "loading_ood": "loading",
    "material_ood": "material",
}
ROUND_DIGITS = {
    "geometry": {name: 2 for name in PARAMETER_GROUPS["geometry"]},
    "loading": {"A": 3, "Ts": 3},
    "material": {"mu": 2},
}
SIDES = ("lower", "upper")
SAFETY_MARGIN_MM = 10.0


@dataclass(frozen=True)
class GenerationConfig:
    sample_count_per_group: int = 50
    seed: int = 20260809
    outside_fraction: float = 0.1
    first_case_number: int = 1001

    def __post_init__(self) -> None:
        if self.sample_count_per_group <= 0:
            raise ValueError("sample_count_per_group must be positive")
        if not 0.0 < self.outside_fraction <= 1.0:
            raise ValueError("outside_fraction must be in (0, 1]")
        if self.first_case_number <= 0:
            raise ValueError("first_case_number must be positive")

    @property
    def total_samples(self) -> int:
        return self.sample_count_per_group * len(OOD_GROUPS)


def _flat_parameters() -> list[tuple[str, str]]:
    return [
        (group, name)
        for group, names in PARAMETER_GROUPS.items()
        for name in names
    ]


def _balanced_assignments(
    names: tuple[str, ...], count: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    side_sizes = {side: count // len(SIDES) for side in SIDES}
    for side in rng.permutation(SIDES)[: count % len(SIDES)]:
        side_sizes[str(side)] += 1

    assignments: list[tuple[str, str]] = []
    for side in SIDES:
        repeats, remainder = divmod(side_sizes[side], len(names))
        selected_names = list(names) * repeats
        if remainder:
            selected_names.extend(rng.permutation(names).tolist()[:remainder])
        assignments.extend((str(name), side) for name in selected_names)
    rng.shuffle(assignments)
    return assignments


def _rounded_sample(unit_values: np.ndarray) -> dict[str, dict[str, float]]:
    sample: dict[str, dict[str, float]] = {
        "geometry": {},
        "loading": {},
        "material": {},
    }
    for unit, (group, name) in zip(unit_values, _flat_parameters()):
        lower, upper = TRAINING_BOUNDS[group][name]
        value = lower + float(unit) * (upper - lower)
        sample[group][name] = round(value, ROUND_DIGITS[group][name])
    return sample


def _outside_value(
    group: str,
    name: str,
    side: str,
    unit_value: float,
    config: GenerationConfig,
) -> float:
    lower, upper = TRAINING_BOUNDS[group][name]
    width = upper - lower
    fraction = max(float(unit_value), 1.0e-6) * config.outside_fraction
    value = lower - fraction * width if side == "lower" else upper + fraction * width
    return round(value, ROUND_DIGITS[group][name])


def _outside_names(sample: dict) -> list[str]:
    outside = []
    for group, names in PARAMETER_GROUPS.items():
        for name in names:
            lower, upper = TRAINING_BOUNDS[group][name]
            value = float(sample[group][name])
            if value < lower or value > upper:
                outside.append(name)
    return outside


def _safety_margin(sample: dict) -> float:
    return (
        float(sample["geometry"]["sy"])
        - float(sample["geometry"]["b2"])
        - 2.0 * float(sample["loading"]["A"])
    )


def _make_case(
    values: dict[str, dict[str, float]],
    case_number: int,
    ood_group: str,
    parameter: str,
    side: str,
) -> dict:
    parameter_group = OOD_GROUPS[ood_group]
    lower, upper = TRAINING_BOUNDS[parameter_group][parameter]
    value = values[parameter_group][parameter]
    boundary = lower if side == "lower" else upper
    distance = abs(value - boundary) / (upper - lower)
    suffix = f"{case_number:04d}"
    return {
        "case_id": f"Case_{suffix}",
        "geometry": {
            **values["geometry"],
            "part_id": f"Geo_Sample_{suffix}",
        },
        "loading": {
            **values["loading"],
            "part_id": f"Load_Sample_{suffix}",
        },
        "material": {
            **values["material"],
            "part_id": f"Mat_Sample_{suffix}",
        },
        "ood": {
            "group": ood_group,
            "parameter": parameter,
            "side": side,
            "training_lower": lower,
            "training_upper": upper,
            "value": value,
            "normalized_ood_distance": round(distance, 8),
        },
    }


def generate_samples(config: GenerationConfig | None = None) -> list[dict]:
    config = config or GenerationConfig()
    samples: list[dict] = []
    next_case_number = config.first_case_number

    for group_index, (ood_group, parameter_group) in enumerate(OOD_GROUPS.items()):
        assignment_rng = np.random.default_rng(config.seed + group_index * 1009)
        assignments = _balanced_assignments(
            PARAMETER_GROUPS[parameter_group],
            config.sample_count_per_group,
            assignment_rng,
        )
        sampler = qmc.LatinHypercube(
            d=len(_flat_parameters()) + 1,
            seed=config.seed + group_index * 7919,
        )
        candidate_pool = sampler.random(n=max(512, config.sample_count_per_group * 8))
        candidate_index = 0

        for parameter, side in assignments:
            while True:
                if candidate_index >= len(candidate_pool):
                    candidate_pool = sampler.random(n=512)
                    candidate_index = 0
                candidate = candidate_pool[candidate_index]
                candidate_index += 1
                values = _rounded_sample(candidate[:-1])
                values[parameter_group][parameter] = _outside_value(
                    parameter_group,
                    parameter,
                    side,
                    candidate[-1],
                    config,
                )
                if _outside_names(values) != [parameter]:
                    continue
                if _safety_margin(values) + 1.0e-9 < SAFETY_MARGIN_MM:
                    continue
                samples.append(
                    _make_case(
                        values,
                        next_case_number,
                        ood_group,
                        parameter,
                        side,
                    )
                )
                next_case_number += 1
                break

    validate_samples(samples, config)
    return samples


def validate_samples(
    samples: list[dict], config: GenerationConfig | None = None
) -> dict:
    config = config or GenerationConfig()
    if len(samples) != config.total_samples:
        raise ValueError(
            f"expected {config.total_samples} samples, found {len(samples)}"
        )

    expected_ids = [
        f"Case_{number:04d}"
        for number in range(
            config.first_case_number,
            config.first_case_number + config.total_samples,
        )
    ]
    actual_ids = [sample.get("case_id") for sample in samples]
    if actual_ids != expected_ids:
        raise ValueError("case IDs must be unique, ordered, and consecutive")

    group_counts = Counter()
    side_counts = Counter()
    target_counts = Counter()
    for sample in samples:
        ood = sample.get("ood", {})
        ood_group = ood.get("group")
        if ood_group not in OOD_GROUPS:
            raise ValueError(f"invalid OOD group: {ood_group}")
        parameter_group = OOD_GROUPS[ood_group]
        parameter = ood.get("parameter")
        side = ood.get("side")
        if parameter not in PARAMETER_GROUPS[parameter_group] or side not in SIDES:
            raise ValueError(f"invalid OOD target for {sample['case_id']}")

        for group, names in PARAMETER_GROUPS.items():
            for name in names:
                value = sample.get(group, {}).get(name)
                if not isinstance(value, (int, float)) or not isfinite(value):
                    raise ValueError(f"invalid {group}.{name} in {sample['case_id']}")

        outside = _outside_names(sample)
        if outside != [parameter]:
            raise ValueError(
                f"{sample['case_id']} must have exactly one outside parameter"
            )
        value = float(sample[parameter_group][parameter])
        lower, upper = TRAINING_BOUNDS[parameter_group][parameter]
        expected_side = "lower" if value < lower else "upper"
        boundary = lower if expected_side == "lower" else upper
        distance = abs(value - boundary) / (upper - lower)
        if expected_side != side or not 0.0 < distance <= config.outside_fraction + 1e-9:
            raise ValueError(f"invalid OOD distance in {sample['case_id']}")
        if abs(float(ood.get("value")) - value) > 1e-12:
            raise ValueError(f"OOD value metadata mismatch in {sample['case_id']}")
        if abs(float(ood.get("normalized_ood_distance")) - distance) > 1e-7:
            raise ValueError(f"OOD distance metadata mismatch in {sample['case_id']}")
        if _safety_margin(sample) + 1e-9 < SAFETY_MARGIN_MM:
            raise ValueError(f"unsafe geometry in {sample['case_id']}")

        group_counts[ood_group] += 1
        side_counts[(ood_group, side)] += 1
        target_counts[(ood_group, parameter, side)] += 1

    expected_groups = Counter(
        {group: config.sample_count_per_group for group in OOD_GROUPS}
    )
    if group_counts != expected_groups:
        raise ValueError(f"invalid OOD group counts: {dict(group_counts)}")
    for ood_group, parameter_group in OOD_GROUPS.items():
        pair_counts = [
            target_counts[(ood_group, parameter, side)]
            for parameter in PARAMETER_GROUPS[parameter_group]
            for side in SIDES
        ]
        if max(pair_counts) - min(pair_counts) > 1:
            raise ValueError(f"unbalanced OOD targets in {ood_group}")
        group_side_counts = [side_counts[(ood_group, side)] for side in SIDES]
        if max(group_side_counts) - min(group_side_counts) > 1:
            raise ValueError(f"unbalanced OOD sides in {ood_group}")

    return {
        "valid_cases": len(samples),
        "group_counts": dict(sorted(group_counts.items())),
        "side_counts": {
            f"{group}:{side}": count
            for (group, side), count in sorted(side_counts.items())
        },
        "target_counts": {
            f"{group}:{parameter}:{side}": count
            for (group, parameter, side), count in sorted(target_counts.items())
        },
        "minimum_safety_margin_mm": min(_safety_margin(sample) for sample in samples),
    }


def _outside_bands(config: GenerationConfig) -> dict[str, dict[str, dict[str, list[float]]]]:
    bands: dict[str, dict[str, dict[str, list[float]]]] = {}
    for group, parameters in TRAINING_BOUNDS.items():
        bands[group] = {}
        for name, (lower, upper) in parameters.items():
            extension = (upper - lower) * config.outside_fraction
            bands[group][name] = {
                "training": [lower, upper],
                "lower_ood": [lower - extension, lower],
                "upper_ood": [upper, upper + extension],
            }
    return bands


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(text, encoding=encoding, newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_csv(samples: list[dict]) -> str:
    parameter_columns = [
        name for group in PARAMETER_GROUPS.values() for name in group
    ]
    fieldnames = [
        "case_id",
        *parameter_columns,
        "ood_group",
        "ood_parameter",
        "ood_side",
        "normalized_ood_distance",
        "safety_margin_mm",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for sample in samples:
        row = {"case_id": sample["case_id"]}
        for group, names in PARAMETER_GROUPS.items():
            row.update({name: sample[group][name] for name in names})
        row.update(
            {
                "ood_group": sample["ood"]["group"],
                "ood_parameter": sample["ood"]["parameter"],
                "ood_side": sample["ood"]["side"],
                "normalized_ood_distance": sample["ood"][
                    "normalized_ood_distance"
                ],
                "safety_margin_mm": round(_safety_margin(sample), 6),
            }
        )
        writer.writerow(row)
    return stream.getvalue()


def write_dataset_artifacts(
    output_dir: Path,
    samples: list[dict],
    config: GenerationConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Validate and atomically publish the parameter, audit, and summary files."""
    config = config or GenerationConfig()
    validation = validate_samples(samples, config)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "parameters": output_dir / "4_Combined_Master_Dataset.json",
        "audit": output_dir / "parameter_audit.csv",
        "summary": output_dir / "dataset_summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing artifacts: "
            + ", ".join(path.name for path in existing)
        )

    metadata = {
        "description": "Grouped out-of-domain dataset for surrogate-model generalization evaluation.",
        "total_samples": len(samples),
        "sampling_method": "Grouped Latin Hypercube Sampling with one strict OOD parameter per case",
        "seed": config.seed,
        "sample_count_per_group": config.sample_count_per_group,
        "outside_fraction": config.outside_fraction,
        "first_case_number": config.first_case_number,
        "safety_constraint": "sy >= b2 + 2*A + 10 mm",
        "parameter_ranges": _outside_bands(config),
    }
    parameters_payload = {
        "dataset_metadata": metadata,
        "parameters_list": samples,
    }
    summary_payload = {
        "dataset_metadata": metadata,
        "validation": validation,
    }

    _atomic_write_text(
        paths["parameters"],
        json.dumps(parameters_payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(paths["audit"], _audit_csv(samples), encoding="utf-8-sig")
    _atomic_write_text(
        paths["summary"],
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
    )
    return paths
