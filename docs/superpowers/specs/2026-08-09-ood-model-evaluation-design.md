# 域外模型测试与预测复用设计

## 目标

基于现有 `evaluation_workspace` 测试框架，评估 MeshGraphNet 和 Transolver 在已完成域外工况上的单步预测效果。域外预测和指标保存到独立目录，并将每个工况的预测保存为可复用 HDF5。

## 数据范围

- 参数工况：`Case_1001` 到 `Case_1150`，共 150 个。
- 有效 HDF5：133 个，以 `ood_generalization_workspace/comsol_hdf5/Case_*.h5` 的有效文件为准。
- 终态失败：17 个，来自 `ood_generalization_workspace/failed_cases.json`。
- 测试只包含 133 个有效工况；17 个失败工况必须写入运行配置，不得静默忽略或重新计算。
- 参数与域外标签来自 `ood_generalization_workspace/4_Combined_Master_Dataset.json` 和 `parameter_audit.csv`。

## 模型与归一化

- 模型：`meshgraphnet`、`transolver`。
- 实验：`n=100, seed=42`。
- checkpoint 仍来自 `training_workspace/runs/<model>/n0100/seed_42/checkpoints/best.pt`。
- 输入、输出和增量还原继续使用训练时冻结的归一化统计量；不得使用域外数据重新拟合均值或标准差。

## 代码结构

新增 `evaluation_workspace/test_ood.py` 作为独立入口。它负责：

1. 扫描并验证域外 HDF5；
2. 校验有效工况与失败工况互斥，并覆盖全部 150 个参数工况；
3. 使用域外数据根目录、参数 JSON 和有效 case ID 创建 `FpcDataset`；
4. 加载原训练 checkpoint；
5. 复用 `materialize_test_predictions()` 生成或跳过预测 HDF5；
6. 复用 `analyze_prediction_directory()`、`write_evaluation_tables()` 和 `plot_model_comparison()` 输出现有分层指标。

`evaluation_workspace/common.py` 增加一个可复用的显式数据源上下文入口。域内 `load_evaluation_context()` 的 manifest 快照校验保持不变；OOD 入口不得伪造域内 split manifest。

## 输出目录

所有结果保存到：

```text
evaluation_workspace/results/ood/n0100_seed42/
├── predictions/
│   ├── meshgraphnet/Case_XXXX.h5
│   └── transolver/Case_XXXX.h5
├── summary.json
├── summary.csv
├── case_metrics.csv
├── time_metrics.csv
├── case_time_metrics.csv
├── extrema.csv
├── percentiles.csv
├── model_comparison.png
├── ood_cases.csv
└── run_config.json
```

`ood_cases.csv`保存每个有效工况的域外分组、目标参数、上下侧和审计信息，便于后续按`geometry_ood`、`loading_ood`、`material_ood`分组统计。

## HDF5复用规则

- 每个模型和工况保存一个预测 HDF5，格式与域内测试完全一致。
- 文件包含绝对 `p/T` 预测、GT、动态节点位置与速度、时间索引、静态拓扑、区域以及归一化统计量。
- 仅当模型名、case ID、checkpoint SHA256和时间步数量都匹配时才复用。
- `REUSE_PREDICTIONS=True`且文件完整时跳过推理；`OVERWRITE_PREDICTIONS=True`时强制重算。
- 域内和域外结果根目录不同，即使case ID未来发生变化也不得相互覆盖。

## 运行配置

`test_ood.py`末尾提供可直接修改的变量，不依赖命令行参数：

```python
MODELS = ["meshgraphnet", "transolver"]
TRAIN_SIZE = 100
SEED = 42
DEVICE = "auto"
REUSE_PREDICTIONS = True
OVERWRITE_PREDICTIONS = False
RELATIVE_ERROR_THRESHOLD_RATIO = 0.01
OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "ood"
```

`run_config.json`还必须记录参数总数、有效数、失败数、失败case ID、HDF5根目录、参数JSON、审计CSV和checkpoint哈希。

## 验证

1. 使用临时域外目录测试133/17式的数据选择逻辑；
2. 验证缺失但未登记失败、有效与失败重叠、无效HDF5都会报错；
3. 验证上下文使用域外case ID和参数路径，同时仍加载原训练checkpoint与归一化；
4. 验证第二次运行能够复用预测HDF5；
5. 运行现有evaluation回归测试；
6. 检查结果只写入`results/ood`，且不修改用户的`DRY_RUN=False`配置。
