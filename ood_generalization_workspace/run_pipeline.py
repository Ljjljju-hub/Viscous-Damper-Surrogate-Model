"""Generate, validate, and calculate the isolated OOD COMSOL dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from ood_generalization_workspace.parameter_generator import (
    GenerationConfig,
    generate_samples,
    validate_samples,
    write_dataset_artifacts,
)


PROJECT_ROOT = Path(__file__).parents[1].resolve()
CONTROLLER_PATH = PROJECT_ROOT / "计算有限元数据" / "run_remaining.py"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "计算有限元数据" / "standard_model.mph"


@dataclass(frozen=True)
class PipelineConfig:
    workspace_root: Path
    model_path: Path
    generation: GenerationConfig = GenerationConfig()
    generate_parameters: bool = True
    overwrite_parameters: bool = False
    dry_run: bool = True
    batch_size: int = 10
    cores: int = 16
    max_retries: int = 2
    pause_seconds: float = 10.0
    timeout_minutes: float = 0.0
    show_worker_window: bool = True
    python_executable: str = sys.executable

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).resolve())
        object.__setattr__(self, "model_path", Path(self.model_path).resolve())
        if self.batch_size <= 0 or self.cores <= 0:
            raise ValueError("batch_size and cores must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.pause_seconds < 0 or self.timeout_minutes < 0:
            raise ValueError("pause_seconds and timeout_minutes cannot be negative")


def validate_workspace_parameters(
    workspace_root: Path,
    generation: GenerationConfig | None = None,
) -> dict:
    generation = generation or GenerationConfig()
    parameter_path = Path(workspace_root).resolve() / "4_Combined_Master_Dataset.json"
    if not parameter_path.is_file():
        raise FileNotFoundError(f"OOD parameter JSON not found: {parameter_path}")
    try:
        payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read OOD parameter JSON: {parameter_path}") from error
    samples = payload.get("parameters_list")
    if not isinstance(samples, list):
        raise ValueError(f"parameters_list must be a list: {parameter_path}")
    return validate_samples(samples, generation)


def prepare_workspace_parameters(
    workspace_root: Path,
    generation: GenerationConfig | None = None,
    *,
    generate: bool,
    overwrite: bool,
) -> dict:
    generation = generation or GenerationConfig()
    workspace_root = Path(workspace_root).resolve()
    parameter_path = workspace_root / "4_Combined_Master_Dataset.json"

    if parameter_path.exists() and not overwrite:
        return validate_workspace_parameters(workspace_root, generation)
    if not generate:
        if parameter_path.exists():
            return validate_workspace_parameters(workspace_root, generation)
        raise FileNotFoundError(
            "OOD parameter JSON is missing and GENERATE_PARAMETERS is False"
        )

    samples = generate_samples(generation)
    write_dataset_artifacts(
        workspace_root,
        samples,
        generation,
        overwrite=overwrite,
    )
    return validate_samples(samples, generation)


def build_controller_command(config: PipelineConfig) -> list[str]:
    command = [
        config.python_executable,
        str(CONTROLLER_PATH),
        "--workspace-root",
        str(config.workspace_root),
        "--model-path",
        str(config.model_path),
        "--batch-size",
        str(config.batch_size),
        "--cores",
        str(config.cores),
        "--max-retries",
        str(config.max_retries),
        "--pause-seconds",
        str(config.pause_seconds),
        "--timeout-minutes",
        str(config.timeout_minutes),
    ]
    if config.dry_run:
        command.append("--dry-run")
    if not config.show_worker_window:
        command.append("--no-worker-window")
    return command


def run_pipeline(config: PipelineConfig) -> int:
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    if not CONTROLLER_PATH.is_file():
        raise FileNotFoundError(f"COMSOL controller not found: {CONTROLLER_PATH}")
    if not config.model_path.is_file():
        raise FileNotFoundError(f"COMSOL model not found: {config.model_path}")

    summary = prepare_workspace_parameters(
        config.workspace_root,
        config.generation,
        generate=config.generate_parameters,
        overwrite=config.overwrite_parameters,
    )
    print(
        "OOD parameters valid: "
        f"cases={summary['valid_cases']} groups={summary['group_counts']}"
    )
    command = build_controller_command(config)
    print("Controller:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=config.workspace_root,
        check=False,
    )
    return completed.returncode


def main() -> int:
    generation = GenerationConfig(
        sample_count_per_group=SAMPLE_COUNT_PER_GROUP,
        seed=RANDOM_SEED,
        outside_fraction=OUTSIDE_FRACTION,
        first_case_number=FIRST_CASE_NUMBER,
    )
    config = PipelineConfig(
        workspace_root=WORKSPACE_ROOT,
        model_path=MODEL_PATH,
        generation=generation,
        generate_parameters=GENERATE_PARAMETERS,
        overwrite_parameters=OVERWRITE_PARAMETERS,
        dry_run=DRY_RUN,
        batch_size=BATCH_SIZE,
        cores=CORES,
        max_retries=MAX_RETRIES,
        pause_seconds=PAUSE_SECONDS,
        timeout_minutes=TIMEOUT_MINUTES,
        show_worker_window=SHOW_WORKER_WINDOW,
    )
    return run_pipeline(config)


# ======================== OOD parameter dataset ========================
WORKSPACE_ROOT = Path(__file__).parent.resolve()
MODEL_PATH = DEFAULT_MODEL_PATH
SAMPLE_COUNT_PER_GROUP = 50  # geometry/loading/material each contain 50 cases
RANDOM_SEED = 20260809       # changing this creates a different OOD dataset
OUTSIDE_FRACTION = 0.10      # extension relative to each training interval width
FIRST_CASE_NUMBER = 1001     # Case_1001 through Case_1150
GENERATE_PARAMETERS = True   # generate missing artifacts before calculation
OVERWRITE_PARAMETERS = False  # True intentionally replaces existing parameters

# ======================== COMSOL execution ========================
DRY_RUN = True               # True validates and prints batches without COMSOL
BATCH_SIZE = 10              # cases per isolated COMSOL process
CORES = 16                   # processor cores used by one COMSOL session
MAX_RETRIES = 2              # extra passes for non-terminal process failures
PAUSE_SECONDS = 10.0         # pause after a worker exits before the next batch
TIMEOUT_MINUTES = 0.0        # 0 disables per-batch timeout
SHOW_WORKER_WINDOW = True    # False runs each worker in a hidden window


if __name__ == "__main__":
    raise SystemExit(main())
