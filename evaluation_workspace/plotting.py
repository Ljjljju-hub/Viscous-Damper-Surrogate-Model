from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_model_comparison(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    models = [row["model"] for row in summary_rows]
    panels = [
        ("normalized_mse", "Normalized MSE"),
        ("p_rmse", "Pressure RMSE (Pa)"),
        ("T_rmse", "Temperature RMSE (K)"),
        ("p_mae", "Pressure MAE (Pa)"),
        ("T_mae", "Temperature MAE (K)"),
        ("p_p95_absolute_error", "Pressure P95 abs. error (Pa)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    colors = ["#2878B5", "#C82423"]
    for axis, (key, title) in zip(axes.flat, panels):
        axis.bar(models, [float(row[key]) for row in summary_rows], color=colors[: len(models)])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_vs_time(step_rows: list[dict], output_path: Path) -> None:
    if not step_rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for model in sorted({row["model"] for row in step_rows}):
        rows = [row for row in step_rows if row["model"] == model]
        axes[0].plot(
            [float(row["physical_time"]) for row in rows],
            [float(row["p_rmse"]) for row in rows],
            label=model,
        )
        axes[1].plot(
            [float(row["physical_time"]) for row in rows],
            [float(row["T_rmse"]) for row in rows],
            label=model,
        )
    axes[0].set_ylabel("Pressure RMSE (Pa)")
    axes[1].set_ylabel("Temperature RMSE (K)")
    axes[1].set_xlabel("Physical time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
