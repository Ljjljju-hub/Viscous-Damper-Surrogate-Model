"""Variable-driven entry point for MeshGraphNet and Transolver training."""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_workspace.run_scale_study import StudyConfig, run_study


def main(
    *,
    models,
    train_sizes,
    seeds,
    epochs,
    batch_size,
    learning_rate,
    early_stopping_patience,
    early_stopping_min_relative_improvement,
    save_every,
    batch_log_every,
    num_workers,
    device,
    meshgraphnet_hidden_size,
    message_passing_steps,
    transolver_hidden_size,
    transolver_layers,
    transolver_heads,
    transolver_slice_num,
    transolver_dropout,
    transolver_mlp_ratio,
    dry_run,
    continue_on_error,
    output_root,
) -> None:
    config = StudyConfig(
        models=tuple(models),
        train_sizes=tuple(train_sizes) if train_sizes is not None else None,
        seeds=tuple(seeds),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_relative_improvement=(
            early_stopping_min_relative_improvement
        ),
        save_every=save_every,
        batch_log_every=batch_log_every,
        num_workers=num_workers,
        device=device,
        meshgraphnet_hidden_size=meshgraphnet_hidden_size,
        message_passing_steps=message_passing_steps,
        transolver_hidden_size=transolver_hidden_size,
        transolver_layers=transolver_layers,
        transolver_heads=transolver_heads,
        transolver_slice_num=transolver_slice_num,
        transolver_dropout=transolver_dropout,
        transolver_mlp_ratio=transolver_mlp_ratio,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
        output_root=Path(output_root),
    )
    run_study(config)


if __name__ == "__main__":
    # ======================== 实验组合 ========================
    # 可选 "meshgraphnet"、"transolver"，也可同时训练两个模型。
    # MODELS = ["meshgraphnet", "transolver"]
    MODELS = ["transolver", "meshgraphnet"]
    # 使用的训练工况数量；设为 None 时自动运行 100、200、...、800。
    # [100, 200, 300, 400, 500, 600, 700, 800]
    TRAIN_SIZES = [100]
    # 重复实验随机种子；每个种子会保存到独立目录。
    SEEDS = [42]

    # ======================== 共享训练参数 ========================
    EPOCHS = 100  # 最大训练轮数
    BATCH_SIZE = 16  # 每个 batch 的图数量
    LEARNING_RATE = 1.0e-4  # Adam 初始学习率
    EARLY_STOPPING_PATIENCE = 10  # 连续多少轮没有显著改善后早停
    # 相对显著改善阈值；0.002 表示验证损失至少下降 0.2% 才重置耐心。
    EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = 0.002
    SAVE_EVERY = 10  # 每隔多少轮额外保存一个 checkpoint
    BATCH_LOG_EVERY = 10  # 每多少个 batch 写一次实时 TensorBoard loss
    NUM_WORKERS = 0  # DataLoader 子进程数；Windows 建议先用 0
    DEVICE = "auto"  # "auto"、"cuda:0" 或 "cpu"

    # ======================== MeshGraphNet 结构 ========================
    MESHGRAPHNET_HIDDEN_SIZE = 128  # 编码器和消息传递层隐藏维度
    MESSAGE_PASSING_STEPS = 15  # Processor 消息传递层数

    # ======================== Transolver 结构 ========================
    TRANSOLVER_HIDDEN_SIZE = 256  # token 隐藏维度
    TRANSOLVER_LAYERS = 8  # Physics-Attention 层数
    TRANSOLVER_HEADS = 8  # 多头注意力 head 数
    TRANSOLVER_SLICE_NUM = 32  # physics-aware slice 数量
    TRANSOLVER_DROPOUT = 0.0  # dropout 概率
    TRANSOLVER_MLP_RATIO = 1  # FFN 隐藏维度倍率

    # ======================== 执行与保存 ========================
    # True 只检查并显示任务；确认无误后改为 False 开始训练。
    DRY_RUN = False
    # False 表示某个任务失败后立即停止；True 表示继续后续组合。
    CONTINUE_ON_ERROR = False
    # 自动保存为 runs/<模型>/n<规模>/seed_<种子>/，无需手动创建子目录。
    OUTPUT_ROOT = WORKSPACE_ROOT / "runs"

    main(
        models=MODELS,
        train_sizes=TRAIN_SIZES,
        seeds=SEEDS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_relative_improvement=(
            EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT
        ),
        save_every=SAVE_EVERY,
        batch_log_every=BATCH_LOG_EVERY,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        meshgraphnet_hidden_size=MESHGRAPHNET_HIDDEN_SIZE,
        message_passing_steps=MESSAGE_PASSING_STEPS,
        transolver_hidden_size=TRANSOLVER_HIDDEN_SIZE,
        transolver_layers=TRANSOLVER_LAYERS,
        transolver_heads=TRANSOLVER_HEADS,
        transolver_slice_num=TRANSOLVER_SLICE_NUM,
        transolver_dropout=TRANSOLVER_DROPOUT,
        transolver_mlp_ratio=TRANSOLVER_MLP_RATIO,
        dry_run=DRY_RUN,
        continue_on_error=CONTINUE_ON_ERROR,
        output_root=OUTPUT_ROOT,
    )
