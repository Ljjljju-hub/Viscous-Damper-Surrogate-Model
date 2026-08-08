import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.experiment_utils import (
    atomic_write_json,
    load_split_manifest,
    verify_manifest_snapshot,
)


MODEL_SCRIPTS = {
    "meshgraphnet": PROJECT_ROOT / "meshGraphNet_self" / "train_worker.py",
    "transolver": PROJECT_ROOT / "transolver_self" / "train_worker.py",
}


@dataclass(frozen=True)
class StudyConfig:
    manifest: Path = WORKSPACE_ROOT / "dataset_split" / "split_manifest.json"
    output_root: Path = WORKSPACE_ROOT / "runs"
    models: Sequence[str] = ("meshgraphnet", "transolver")
    train_sizes: Optional[Sequence[int]] = None
    seeds: Sequence[int] = (42, 43, 44)
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 0
    learning_rate: float = 1.0e-4
    early_stopping_patience: int = 15
    save_every: int = 10
    device: str = "auto"
    meshgraphnet_hidden_size: int = 128
    message_passing_steps: int = 15
    transolver_hidden_size: int = 256
    transolver_layers: int = 8
    transolver_heads: int = 8
    transolver_slice_num: int = 32
    transolver_dropout: float = 0.0
    transolver_mlp_ratio: int = 1
    dry_run: bool = False
    continue_on_error: bool = False
    allow_temporary_manifest: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_sizes(train_pool_count: int) -> list:
    return list(range(100, train_pool_count + 1, 100))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(
    *,
    model_name: str,
    train_size: int,
    seed: int,
    args,
    manifest: dict,
) -> bool:
    run_dir = (
        args.output_root.resolve()
        / model_name
        / f"n{train_size:04d}"
        / f"seed_{seed}"
    )
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    metrics_file = run_dir / "metrics.csv"
    summary_file = run_dir / "summary.json"
    status_file = run_dir / "status.json"
    log_file = run_dir / "train.log"

    request = {
        "version": 1,
        "model": model_name,
        "train_size": train_size,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "save_every": args.save_every,
        "device": args.device,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest.resolve()),
    }
    if model_name == "meshgraphnet":
        request.update(
            {
                "hidden_size": args.meshgraphnet_hidden_size,
                "message_passing_steps": args.message_passing_steps,
            }
        )
    else:
        request.update(
            {
                "hidden_size": args.transolver_hidden_size,
                "layers": args.transolver_layers,
                "heads": args.transolver_heads,
                "slice_num": args.transolver_slice_num,
                "dropout": args.transolver_dropout,
                "mlp_ratio": args.transolver_mlp_ratio,
            }
        )
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("completed"):
            previous_status = (
                json.loads(status_file.read_text(encoding="utf-8"))
                if status_file.exists()
                else {}
            )
            if previous_status.get("request") != request:
                raise RuntimeError(
                    f"Completed run has a different configuration: {run_dir}. "
                    "Set a different OUTPUT_ROOT in training_workspace/train.py "
                    "for the new experiment."
                )
            print(f"SKIP complete: {model_name} n={train_size} seed={seed}")
            return True

    command = [
        sys.executable,
        str(MODEL_SCRIPTS[model_name]),
        "--data-root",
        manifest["data_root"],
        "--parameters-json",
        manifest["parameters_json"],
        "--split-manifest",
        str(args.manifest.resolve()),
        "--train-size",
        str(train_size),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-dir",
        str(tensorboard_dir),
        "--metrics-file",
        str(metrics_file),
        "--summary-file",
        str(summary_file),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--evaluate-test",
    ]
    if model_name == "meshgraphnet":
        command.extend(
            [
                "--hidden-size",
                str(args.meshgraphnet_hidden_size),
                "--message-passing-steps",
                str(args.message_passing_steps),
            ]
        )
    else:
        command.extend(
            [
                "--hidden-size",
                str(args.transolver_hidden_size),
                "--layers",
                str(args.transolver_layers),
                "--heads",
                str(args.transolver_heads),
                "--slice-num",
                str(args.transolver_slice_num),
                "--dropout",
                str(args.transolver_dropout),
                "--mlp-ratio",
                str(args.transolver_mlp_ratio),
            ]
        )
    last_checkpoint = checkpoint_dir / "last.pt"
    resumed = last_checkpoint.exists()
    if resumed:
        command.extend(["--resume", str(last_checkpoint)])

    print(
        f"{'DRY' if args.dry_run else 'RUN'} {model_name} "
        f"n={train_size} seed={seed} resume={resumed}"
    )
    if args.dry_run:
        print(f"  output={run_dir}")
        print(f"  worker={MODEL_SCRIPTS[model_name].name}")
        return True

    run_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "starting",
        "model": model_name,
        "train_size": train_size,
        "seed": seed,
        "started_at_utc": utc_now(),
        "resumed": resumed,
        "request": request,
        "command": command,
    }
    atomic_write_json(status_file, status)

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with log_file.open("a", encoding="utf-8") as log_stream:
        log_stream.write(
            f"\n[{utc_now()}] {subprocess.list2cmdline(command)}\n"
        )
        log_stream.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        status.update({"state": "running", "pid": process.pid})
        atomic_write_json(status_file, status)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_stream.write(line)
                log_stream.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            status.update({"state": "interrupted", "ended_at_utc": utc_now()})
            atomic_write_json(status_file, status)
            raise

    success = return_code == 0 and summary_file.exists()
    status.update(
        {
            "state": "complete" if success else "failed",
            "return_code": return_code,
            "ended_at_utc": utc_now(),
        }
    )
    atomic_write_json(status_file, status)
    return success


def run_study(args: StudyConfig) -> None:
    unknown_models = [model for model in args.models if model not in MODEL_SCRIPTS]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    manifest = load_split_manifest(args.manifest)
    if manifest.get("temporary_incomplete") and not args.allow_temporary_manifest:
        raise RuntimeError(
            "The split manifest was created from incomplete COMSOL data. "
            "Create the final manifest before starting formal training."
        )
    snapshot_errors = verify_manifest_snapshot(manifest)
    if snapshot_errors:
        preview = "\n".join(snapshot_errors[:20])
        raise RuntimeError(f"Frozen dataset snapshot changed:\n{preview}")
    train_pool_count = len(manifest["train_pool"])
    manifest_sizes = [int(size) for size in manifest.get("train_sizes", [])]
    sizes = args.train_sizes or manifest_sizes or default_sizes(train_pool_count)
    invalid_sizes = [size for size in sizes if size < 1 or size > train_pool_count]
    if invalid_sizes:
        raise ValueError(
            f"Invalid train sizes {invalid_sizes}; train pool has {train_pool_count}."
        )

    total = len(args.models) * len(sizes) * len(args.seeds)
    print(
        f"experiments={total} models={args.models} sizes={sizes} "
        f"seeds={args.seeds}"
    )
    failures = []
    for model_name in args.models:
        for train_size in sizes:
            for seed in args.seeds:
                success = run_one(
                    model_name=model_name,
                    train_size=train_size,
                    seed=seed,
                    args=args,
                    manifest=manifest,
                )
                if not success:
                    failures.append((model_name, train_size, seed))
                    if not args.continue_on_error:
                        raise RuntimeError(f"Experiment failed: {failures[-1]}")

    if failures:
        raise RuntimeError(f"{len(failures)} experiments failed: {failures}")
    if args.dry_run:
        print("Dry-run plan is valid; no training was started.")
    else:
        print("All requested experiments are complete.")
