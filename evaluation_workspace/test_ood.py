"""Evaluate trained models on the isolated out-of-domain dataset."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_workspace.common import load_evaluation_context_from_cases
from evaluation_workspace.ood_evaluation import (
    build_ood_inventory,
    write_ood_case_audit,
)
from evaluation_workspace.plotting import plot_model_comparison
from evaluation_workspace.test_pipeline import (
    analyze_prediction_directory,
    materialize_test_predictions,
    write_evaluation_tables,
)
from meshGraphNet_self.experiment_utils import atomic_write_json


def main(
    *,
    models,
    train_size,
    seed,
    device,
    reuse_predictions,
    overwrite_predictions,
    relative_error_threshold_ratio,
    ood_workspace,
    output_root,
) -> None:
    inventory = build_ood_inventory(Path(ood_workspace))
    case_ids = list(inventory.valid_case_ids)
    run_dir = Path(output_root) / f"n{train_size:04d}_seed{seed}"
    prediction_root = run_dir / "predictions"
    context = load_evaluation_context_from_cases(
        models=models,
        train_size=train_size,
        seed=seed,
        device=device,
        data_root=inventory.data_root,
        parameters_json=inventory.parameters_json,
        case_ids=case_ids,
        source_name="ood",
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "out_of_domain",
        "models": list(models),
        "train_size": int(train_size),
        "seed": int(seed),
        "device": str(context.device),
        "reuse_predictions": bool(reuse_predictions),
        "overwrite_predictions": bool(overwrite_predictions),
        "relative_error_threshold_ratio": float(
            relative_error_threshold_ratio
        ),
        "parameter_case_count": len(inventory.parameter_case_ids),
        "evaluated_case_count": len(inventory.valid_case_ids),
        "failed_case_count": len(inventory.failed_case_ids),
        "evaluated_case_ids": case_ids,
        "failed_case_ids": list(inventory.failed_case_ids),
        "ood_workspace": str(inventory.workspace_root),
        "data_root": str(inventory.data_root),
        "parameters_json": str(inventory.parameters_json),
        "parameter_audit_csv": str(inventory.audit_csv),
        "failed_cases_json": str(inventory.failed_cases_json),
        "checkpoint_sha256": dict(context.checkpoint_hashes),
        "output_directory": str(run_dir.resolve()),
    }
    atomic_write_json(run_dir / "run_config.json", config)
    write_ood_case_audit(inventory, run_dir / "ood_cases.csv")

    print(
        f"OOD cases: parameters={len(inventory.parameter_case_ids)} "
        f"evaluate={len(case_ids)} failed={len(inventory.failed_case_ids)}"
    )
    materialize_test_predictions(
        context,
        prediction_root,
        reuse=reuse_predictions,
        overwrite=overwrite_predictions,
        case_ids=case_ids,
    )
    tables = analyze_prediction_directory(
        prediction_root,
        threshold_ratio=relative_error_threshold_ratio,
        models=list(models),
        case_ids=case_ids,
    )
    write_evaluation_tables(tables, run_dir)
    plot_model_comparison(tables.summary, run_dir / "model_comparison.png")

    print(f"OOD evaluation complete: {run_dir.resolve()}")
    for row in tables.summary:
        print(
            f"{row['model']}: normalized_mse={row['normalized_mse']:.6g} "
            f"p_rmse={row['p_rmse']:.6g} Pa T_rmse={row['T_rmse']:.6g} K"
        )


if __name__ == "__main__":
    # ======================== 模型与训练实验 ========================
    MODELS = ["meshgraphnet", "transolver"]
    TRAIN_SIZE = 100
    SEED = 42
    DEVICE = "auto"  # "auto"、"cuda:0" 或 "cpu"

    # ======================== 预测HDF5复用 ========================
    REUSE_PREDICTIONS = True  # checkpoint和时间步匹配时跳过已有预测
    OVERWRITE_PREDICTIONS = False  # True强制重新预测全部有效OOD工况

    # 真值绝对值低于每工况最大绝对真值1%时，不统计单点相对误差。
    RELATIVE_ERROR_THRESHOLD_RATIO = 0.01
    OOD_WORKSPACE = PROJECT_ROOT / "ood_generalization_workspace"
    OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "ood"

    main(
        models=MODELS,
        train_size=TRAIN_SIZE,
        seed=SEED,
        device=DEVICE,
        reuse_predictions=REUSE_PREDICTIONS,
        overwrite_predictions=OVERWRITE_PREDICTIONS,
        relative_error_threshold_ratio=RELATIVE_ERROR_THRESHOLD_RATIO,
        ood_workspace=OOD_WORKSPACE,
        output_root=OUTPUT_ROOT,
    )
