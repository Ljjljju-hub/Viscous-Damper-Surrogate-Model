import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parent
EVALUATE_SCRIPT = WORKSPACE_ROOT / "evaluate_experiment.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable rollouts for every completed scale experiment."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=WORKSPACE_ROOT / "runs",
    )
    parser.add_argument("--rollout-count", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = sorted(args.runs_root.resolve().rglob("summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No completed runs under {args.runs_root}.")
    print(f"completed_runs={len(summaries)}")
    for index, summary in enumerate(summaries, start=1):
        run_dir = summary.parent
        command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--rollout-count",
            str(args.rollout_count),
            "--device",
            args.device,
        ]
        if args.save_predictions:
            command.append("--save-predictions")
        if args.force:
            command.append("--force")
        print(f"EVAL {index}/{len(summaries)} {run_dir}")
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed for {run_dir}.")


if __name__ == "__main__":
    main()
