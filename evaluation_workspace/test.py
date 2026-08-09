"""Run reusable full-test inference and hierarchical metric analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_workspace.common import load_evaluation_context
from evaluation_workspace.plotting import plot_model_comparison
from evaluation_workspace.test_pipeline import (
    analyze_prediction_directory,
    materialize_test_predictions,
    write_evaluation_tables,
)


def main(
    *,
    models,
    train_size,
    seed,
    device,
    reuse_predictions,
    overwrite_predictions,
    relative_error_threshold_ratio,
    output_root,
) -> None:
    run_dir = Path(output_root) / f"n{train_size:04d}_seed{seed}"
    prediction_root = run_dir / "predictions"
    context = load_evaluation_context(
        models=models,
        train_size=train_size,
        seed=seed,
        device=device,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "models": list(models),
        "train_size": train_size,
        "seed": seed,
        "device": str(context.device),
        "reuse_predictions": reuse_predictions,
        "overwrite_predictions": overwrite_predictions,
        "relative_error_threshold_ratio": relative_error_threshold_ratio,
        "test_case_count": len(context.manifest["test"]),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    materialize_test_predictions(
        context,
        prediction_root,
        reuse=reuse_predictions,
        overwrite=overwrite_predictions,
    )
    tables = analyze_prediction_directory(
        prediction_root,
        threshold_ratio=relative_error_threshold_ratio,
        models=list(models),
        case_ids=list(context.manifest["test"]),
    )
    write_evaluation_tables(tables, run_dir)
    plot_model_comparison(tables.summary, run_dir / "model_comparison.png")
    print(f"Test evaluation complete: {run_dir.resolve()}")
    for row in tables.summary:
        print(
            f"{row['model']}: normalized_mse={row['normalized_mse']:.6g} "
            f"p_rmse={row['p_rmse']:.6g} Pa T_rmse={row['T_rmse']:.6g} K"
        )


if __name__ == "__main__":
    # ======================== 模型与训练实验 ========================
    MODELS = ["meshgraphnet", "transolver"]  # 可只保留其中一个模型
    TRAIN_SIZE = 100  # 对应 runs/<模型>/n0100
    SEED = 42  # 对应 runs/<模型>/n0100/seed_42
    DEVICE = "auto"  # "auto"、"cuda:0" 或 "cpu"

    # ======================== 预测复用 ========================
    REUSE_PREDICTIONS = True  # checkpoint 未变化时跳过已有完整预测
    OVERWRITE_PREDICTIONS = False  # True 强制重新生成全部预测

    # 真值绝对值低于该工况最大绝对真值的 1% 时，不计算点相对误差。
    RELATIVE_ERROR_THRESHOLD_RATIO = 0.01
    OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "test"

    main(
        models=MODELS,
        train_size=TRAIN_SIZE,
        seed=SEED,
        device=DEVICE,
        reuse_predictions=REUSE_PREDICTIONS,
        overwrite_predictions=OVERWRITE_PREDICTIONS,
        relative_error_threshold_ratio=RELATIVE_ERROR_THRESHOLD_RATIO,
        output_root=OUTPUT_ROOT,
    )
