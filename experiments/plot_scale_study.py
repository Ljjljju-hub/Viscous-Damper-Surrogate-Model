import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_METRICS = (
    "best_valid_normalized_mse",
    "test_normalized_mse",
    "test_rmse_p",
    "test_rmse_T",
    "total_epoch_seconds",
    "best_epoch",
)
ROLLOUT_METRICS = ("rollout_rmse_p", "rollout_rmse_T")
MODEL_LABELS = {"meshgraphnet": "MeshGraphNet", "transolver": "Transolver"}


def read_csv(path: Path) -> list:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(runs_root: Path) -> list:
    runs = []
    for summary_path in sorted(runs_root.rglob("summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "model_name": summary["model_name"],
            "train_size": int(summary["train_size"]),
            "seed": int(summary["seed"]),
            "run_dir": str(summary_path.parent),
        }
        for metric in SUMMARY_METRICS:
            if summary.get(metric) is not None:
                row[metric] = float(summary[metric])
        evaluation_path = summary_path.parent / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation.get("completed"):
                for metric in ROLLOUT_METRICS:
                    row[metric] = float(evaluation[metric])
        runs.append(row)
    return runs


def aggregate_runs(runs: list) -> list:
    groups = defaultdict(list)
    for row in runs:
        groups[(row["model_name"], row["train_size"])].append(row)
    aggregated = []
    for (model_name, train_size), group in sorted(groups.items()):
        result = {
            "model_name": model_name,
            "train_size": train_size,
            "run_count": len(group),
        }
        for metric in (*SUMMARY_METRICS, *ROLLOUT_METRICS):
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                result[f"{metric}_mean"] = statistics.fmean(values)
                result[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
                result[f"{metric}_count"] = len(values)
        aggregated.append(result)
    return aggregated


def plot_metric(aggregated: list, metric: str, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for model_name, label in MODEL_LABELS.items():
        rows = [
            row
            for row in aggregated
            if row["model_name"] == model_name and f"{metric}_mean" in row
        ]
        rows.sort(key=lambda row: int(row["train_size"]))
        if not rows:
            continue
        axis.errorbar(
            [int(row["train_size"]) for row in rows],
            [float(row[f"{metric}_mean"]) for row in rows],
            yerr=[float(row[f"{metric}_std"]) for row in rows],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=label,
        )
        plotted = True
    if plotted:
        axis.set_xlabel("Training cases")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{metric}.{suffix}", dpi=220)
    plt.close(fig)


def plot_training_curves(runs: list, output_dir: Path) -> None:
    grouped = defaultdict(list)
    for run in runs:
        metrics_path = Path(run["run_dir"]) / "metrics.csv"
        if metrics_path.exists():
            grouped[(run["model_name"], run["train_size"])].append(
                read_csv(metrics_path)
            )
    for model_name, label in MODEL_LABELS.items():
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        plotted = False
        sizes = sorted(
            size for current_model, size in grouped if current_model == model_name
        )
        for train_size in sizes:
            by_epoch = defaultdict(list)
            for history in grouped[(model_name, train_size)]:
                for row in history:
                    by_epoch[int(row["epoch"])].append(
                        float(row["valid_normalized_mse"])
                    )
            epochs = sorted(by_epoch)
            axis.plot(
                epochs,
                [statistics.fmean(by_epoch[epoch]) for epoch in epochs],
                label=f"N={train_size}",
            )
            plotted = True
        if plotted:
            axis.set_title(label)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Validation normalized MSE")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(
                    output_dir / f"training_curves_{model_name}.{suffix}",
                    dpi=220,
                )
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate repeated scale experiments and generate curves."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "dataset_scale" / "runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "dataset_scale" / "plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.runs_root.resolve())
    if not runs:
        raise FileNotFoundError(f"No completed summaries under {args.runs_root}.")
    aggregated = aggregate_runs(runs)
    write_csv(output_dir / "individual_results.csv", runs)
    write_csv(output_dir / "aggregate.csv", aggregated)
    for metric in (
        "best_valid_normalized_mse",
        "test_normalized_mse",
        "test_rmse_p",
        "test_rmse_T",
        "rollout_rmse_p",
        "rollout_rmse_T",
        "total_epoch_seconds",
    ):
        plot_metric(aggregated, metric, output_dir)
    plot_training_curves(runs, output_dir)
    print(f"runs={len(runs)} groups={len(aggregated)} output={output_dir}")


if __name__ == "__main__":
    main()
