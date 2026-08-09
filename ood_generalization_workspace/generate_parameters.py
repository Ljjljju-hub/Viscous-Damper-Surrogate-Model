"""Generate the fixed grouped OOD parameter dataset."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from ood_generalization_workspace.parameter_generator import (
    GenerationConfig,
    generate_samples,
    write_dataset_artifacts,
)


WORKSPACE_ROOT = Path(__file__).parent.resolve()


def main() -> int:
    config = GenerationConfig(
        sample_count_per_group=SAMPLE_COUNT_PER_GROUP,
        seed=RANDOM_SEED,
        outside_fraction=OUTSIDE_FRACTION,
        first_case_number=FIRST_CASE_NUMBER,
    )
    samples = generate_samples(config)
    paths = write_dataset_artifacts(
        WORKSPACE_ROOT,
        samples,
        config,
        overwrite=OVERWRITE,
    )
    print(
        f"Generated {len(samples)} OOD cases: "
        f"{samples[0]['case_id']}..{samples[-1]['case_id']}"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


# ======================== Parameter generation ========================
SAMPLE_COUNT_PER_GROUP = 50  # geometry/loading/material each contain 50 cases
RANDOM_SEED = 20260809       # fixed seed for reproducible parameters
OUTSIDE_FRACTION = 0.10      # OOD band width / original training interval width
FIRST_CASE_NUMBER = 1001     # produces Case_1001 through Case_1150
OVERWRITE = False            # True deliberately replaces all three artifacts


if __name__ == "__main__":
    raise SystemExit(main())
