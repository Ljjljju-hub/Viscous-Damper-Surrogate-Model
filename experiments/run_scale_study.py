import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.experiment_utils import (
    atomic_write_json,
    load_split_manifest,
    verify_manifest_snapshot,
)


MODEL_SCRIPTS = {
    "meshgraphnet": PROJECT_ROOT / "meshGraphNet_self" / "train.py",
    "transolver": PROJECT_ROOT / "transolver_self" / "train.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_sizes(train_pool_count: int) -> list:
    return list(range(100, train_pool_count + 1, 100))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable MeshGraphNet/Transolver dataset-scale experiments."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "dataset_scale"
        / "split_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "dataset_scale" / "runs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SCRIPTS),
        default=list(MODEL_SCRIPTS),
    )
    parser.add_argument("--train-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-temporary-manifest", action="store_true")
    return parser.parse_args()


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

    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("completed"):
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
    last_checkpoint = checkpoint_dir / "last.pt"
    resumed = last_checkpoint.exists()
    if resumed:
        command.extend(["--resume", str(last_checkpoint)])

    print(
        f"{'DRY' if args.dry_run else 'RUN'} {model_name} "
        f"n={train_size} seed={seed} resume={resumed}"
    )
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return True

    run_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "starting",
        "model": model_name,
        "train_size": train_size,
        "seed": seed,
        "started_at_utc": utc_now(),
        "resumed": resumed,
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


def main():
    args = parse_args()
    manifest = load_split_manifest(args.manifest)
    if manifest.get("temporary_incomplete") and not args.allow_temporary_manifest:
        raise RuntimeError(
            "The split manifest was created from incomplete COMSOL data. "
            "Create the final manifest or pass --allow-temporary-manifest for testing."
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
    print("All requested experiments are complete.")


if __name__ == "__main__":
    main()
