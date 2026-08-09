"""Export a selected test case as a ParaView temporal comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_workspace.common import load_evaluation_context
from evaluation_workspace.plotting import plot_error_vs_time
from evaluation_workspace.representative_cases import (
    export_representative_cases,
    select_representative_cases,
)
from evaluation_workspace.visualization_pipeline import (
    build_step_metric_rows,
    load_saved_one_step_sequence,
    rollout_sequence,
    write_step_metrics,
)
from evaluation_workspace.vtu_export import export_comparison_pvd


def main(
    *,
    models,
    train_size,
    seed,
    case_id,
    start_index,
    steps,
    source_mode,
    case_selection_mode,
    device,
    threshold_ratio,
    test_result_root,
    visualization_root,
) -> None:
    if source_mode not in {"saved_one_step", "rollout"}:
        raise ValueError("SOURCE_MODE must be 'saved_one_step' or 'rollout'.")
    if case_selection_mode not in {"manual", "test_extremes"}:
        raise ValueError(
            "CASE_SELECTION_MODE must be 'manual' or 'test_extremes'."
        )
    context = load_evaluation_context(
        models=models,
        train_size=train_size,
        seed=seed,
        device=device,
    )
    test_run_dir = Path(test_result_root) / f"n{train_size:04d}_seed{seed}"
    prediction_root = test_run_dir / "predictions"
    if case_selection_mode == "test_extremes":
        selections = select_representative_cases(
            test_run_dir / "case_metrics.csv", list(models)
        )
        output_dir = (
            Path(visualization_root)
            / "test_extremes"
            / f"n{train_size:04d}_seed{seed}"
        )
        exported = export_representative_cases(
            context,
            selections,
            prediction_root,
            output_dir,
            threshold_ratio,
        )
        print(
            f"Representative export complete: {len(exported)} unique cases, "
            f"{(output_dir / 'representative_cases.csv').resolve()}"
        )
        return

    if source_mode == "saved_one_step":
        sequence = load_saved_one_step_sequence(
            context, prediction_root, case_id, start_index, steps
        )
    else:
        sequence = rollout_sequence(context, case_id, start_index, steps)
    output_dir = (
        Path(visualization_root)
        / case_id
        / f"start_{start_index:04d}"
        / source_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "models": list(models),
        "train_size": train_size,
        "seed": seed,
        "case_id": case_id,
        "start_index": start_index,
        "steps": steps,
        "source_mode": source_mode,
        "case_selection_mode": case_selection_mode,
        "device": str(context.device),
        "relative_error_threshold_ratio": threshold_ratio,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = build_step_metric_rows(sequence, threshold_ratio)
    write_step_metrics(rows, output_dir / "step_metrics.csv")
    plot_error_vs_time(rows, output_dir / "error_vs_time.png")
    pvd_path = export_comparison_pvd(sequence, output_dir, threshold_ratio)
    print(f"Time-series visualization complete: {pvd_path.resolve()}")


if __name__ == "__main__":
    # ======================== 模型与工况 ========================
    MODELS = ["meshgraphnet", "transolver"]
    TRAIN_SIZE = 100
    SEED = 42
    CASE_ID = "Case_0866"  # test split 中需要分析的工况

    # ======================== 时序范围 ========================
    START_INDEX = 0  # 已知初始场的时间索引
    STEPS = None  # None 表示一直预测到该工况末尾
    SOURCE_MODE = "saved_one_step"  # "saved_one_step" 或 "rollout"
    # "manual" 导出上面的 CASE_ID；"test_extremes" 自动导出指标极值工况。
    CASE_SELECTION_MODE = "test_extremes"
    DEVICE = "auto"
    RELATIVE_ERROR_THRESHOLD_RATIO = 0.01

    TEST_RESULT_ROOT = WORKSPACE_ROOT / "results" / "test"
    VISUALIZATION_ROOT = WORKSPACE_ROOT / "results" / "visualization"

    main(
        models=MODELS,
        train_size=TRAIN_SIZE,
        seed=SEED,
        case_id=CASE_ID,
        start_index=START_INDEX,
        steps=STEPS,
        source_mode=SOURCE_MODE,
        case_selection_mode=CASE_SELECTION_MODE,
        device=DEVICE,
        threshold_ratio=RELATIVE_ERROR_THRESHOLD_RATIO,
        test_result_root=TEST_RESULT_ROOT,
        visualization_root=VISUALIZATION_ROOT,
    )
