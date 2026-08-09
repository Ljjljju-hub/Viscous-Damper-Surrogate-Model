# 模型测试与时序可视化设计

## 1. 目标

为已经训练完成的 MeshGraphNet 和 Transolver 建立独立评估工作区，提供两条互不耦合的人工入口：

1. `test.py`：运行完整 test split 的单步预测，持久化绝对预测场，并生成分层指标。
2. `visualize_timeseries.py`：读取已保存的单步预测，或从指定初始时刻执行自回归 rollout，导出 ParaView 时序文件。

两条流程可以独立运行。测试预测一旦完成，指标重算和单步可视化不再调用神经网络。

## 2. 目录结构

```text
evaluation_workspace/
├── test.py
├── visualize_timeseries.py
├── common.py
├── metrics.py
├── prediction_store.py
├── vtu_export.py
├── plotting.py
├── tests/
├── README.md
└── results/
```

人工只修改两个入口文件末尾的变量区，不使用命令行参数。

## 3. `test.py` 单步测试

### 3.1 样本定义

test split 完全取自冻结的 `training_workspace/dataset_split/split_manifest.json`。对每个 HDF5 的相邻时间点：

```text
真实当前场 y_k -> 模型预测绝对下一时刻场 y_hat_(k+1)
真实下一时刻场 y_(k+1) -> 误差统计
```

每一步都重新输入真实当前场，因此这是 teacher-forced 单步测试，不是自回归 rollout。

### 3.2 配置

入口变量包括：

```python
MODELS = ["meshgraphnet", "transolver"]
TRAIN_SIZE = 100
SEED = 42
DEVICE = "auto"
REUSE_PREDICTIONS = True
OVERWRITE_PREDICTIONS = False
RELATIVE_ERROR_THRESHOLD_RATIO = 0.01
```

checkpoint 默认从 `training_workspace/runs/<model>/n0100/seed_42/checkpoints/best.pt` 读取。

### 3.3 预测结果持久化

每个模型、每个测试工况写一个独立 HDF5：

```text
results/test/n0100_seed42/predictions/
├── meshgraphnet/Case_XXXX.h5
└── transolver/Case_XXXX.h5
```

文件结构：

```text
attrs/model_name
attrs/case_id
attrs/checkpoint_path
attrs/checkpoint_sha256
attrs/prediction_mode = "one_step"
attrs/complete = true
time_indices                 [K]
time_steps                   [K]
mesh/positions               [K, N, 2]
mesh/velocity                [K, N, 2]
mesh/face                    [3, F]
mesh/region                  [N]
truth/p                      [K, N]
truth/T                      [K, N]
prediction/p                 [K, N]
prediction/T                 [K, N]
normalization/output_mean    [2]
normalization/output_std     [2]
```

其中 `K` 是预测目标时刻数量，通常为 150；位置和速度对应目标时刻的实时动网格。数组使用 float32，拓扑和区域使用整数。

预测文件先写 `.partial.h5`，所有数据完整并通过形状检查后设置 `complete=true`，再原子替换正式文件。启用复用时，只有 checkpoint SHA256、模型、工况和数据形状均匹配才跳过推理；中断后重新运行会继续缺失工况。

## 4. 指标定义

令绝对预测误差为：

```text
e_p = p_prediction - p_truth
e_T = T_prediction - T_truth
```

每个字段统计：

```text
RMSE = sqrt(sum(e^2) / count)
MAE = sum(abs(e)) / count
P95 absolute error
P99 absolute error
maximum absolute error
maximum valid pointwise relative error
```

`normalized_mse` 使用训练 checkpoint 内冻结的输出增量标准差：

```text
normalized_mse =
    (sum((e_p / sigma_delta_p)^2) + sum((e_T / sigma_delta_T)^2))
    / (2 * node_time_count)
```

### 4.1 相对误差近零规则

对每个工况和字段，先在该工况全部目标时间与节点中计算：

```text
threshold = 0.01 * max(abs(truth))
```

只有 `abs(truth) >= threshold` 的节点时刻参与点相对误差：

```text
relative_error_percent = abs(prediction - truth) / abs(truth) * 100
```

RMSE、MAE、绝对误差分位数和最大绝对误差始终使用全部节点。CSV 同时记录相对误差有效点数、排除点数和实际阈值。

## 5. 分层聚合

总体 RMSE 不对各工况 RMSE 做算术平均，而是汇总全部平方误差后只开一次根号。

`test.py` 生成：

```text
summary.csv
case_metrics.csv
time_metrics.csv
case_time_metrics.csv
extrema.csv
percentiles.csv
summary.json
model_comparison.png
```

各层含义：

1. `summary.csv`：每个模型一行，汇总全部测试 HDF5、全部时间步和全部节点。
2. `case_metrics.csv`：每个模型、每个 HDF5 一行，汇总该工况全部时间步和节点。
3. `time_metrics.csv`：每个模型、统一时间索引一行，汇总该时间步的全部测试工况和节点。
4. `case_time_metrics.csv`：每个模型、每个 HDF5、每个时间步一行，只汇总当前网格的节点。
5. `extrema.csv`：记录全局及每工况最大绝对/有效相对误差点的 model、field、case、time index、物理时间、node index、动态坐标、真值、预测值和误差。
6. `percentiles.csv`：记录全局和每工况的 P95、P99、Max 绝对误差。

对于具有相同时间索引但物理时间不同的工况，`time_metrics.csv` 同时记录该索引下的最小、最大和平均物理时间，不把它描述为唯一物理时刻。

## 6. `visualize_timeseries.py` 时序可视化

### 6.1 配置

```python
MODELS = ["meshgraphnet", "transolver"]
CASE_ID = "Case_0866"
START_INDEX = 0
STEPS = None
SOURCE_MODE = "saved_one_step"  # 或 "rollout"
DEVICE = "auto"
```

`saved_one_step` 只读取 `test.py` 的预测 HDF5，不加载模型。`rollout` 在起点使用真实场，后续输入前一步预测，并加载所选模型的 `best.pt`。

`START_INDEX` 表示已知初始场索引，`STEPS` 表示从该索引向后预测的转移次数。导出的第 0 帧是初始真实场；因为它是两个模型共同获得的初始条件，模型字段写入同一个真实值且误差为 0。随后第 1 帧对应 `START_INDEX + 1` 的预测结果。

### 6.2 ParaView 输出

```text
results/visualization/Case_0866/start_0000/<source_mode>/
├── comparison.pvd
├── frames/frame_0000.vtu
├── frames/frame_0001.vtu
├── ...
├── step_metrics.csv
└── error_vs_time.png
```

PVD 是时间索引文件，必须与 `frames/*.vtu` 一起保留。每帧 VTU 使用该时刻的动态节点位置和静态拓扑，节点字段包括：

```text
p_ground_truth
T_ground_truth
p_<model>
T_<model>
p_abs_error_<model>
T_abs_error_<model>
p_relative_error_<model>
T_relative_error_<model>
relative_error_valid_p
relative_error_valid_T
mesh_region
mesh_velocity
```

低于相对误差阈值的点写入 NaN，并由有效掩码字段明确标记。`step_metrics.csv` 对每个模型、每帧记录 RMSE、MAE、最大绝对误差、最大有效相对误差和累计 rollout 步数。

## 7. 错误处理与可重复性

1. checkpoint 的 `model_name`、`model_config`、归一化状态和目标字段必须完整。
2. 两个模型必须使用同一个 split manifest、训练规模和测试 case 列表。
3. 预测 HDF5 记录 checkpoint SHA256，checkpoint 改变后不得静默复用旧预测。
4. 指标文件通过临时文件原子替换；预测中断不会产生伪完整 HDF5。
5. 预测 HDF5 保存 checkpoint 中冻结的输出均值和标准差；test 统计只读取预测 HDF5，不重新加载模型或拟合归一化器。

## 8. 验证

自动测试覆盖：

1. 跨 batch/case/time 的流式 RMSE 等于直接展开数组后的 RMSE。
2. 总体 RMSE 不是简单平均各 case RMSE。
3. 1% 相对误差阈值正确排除近零真值并记录数量。
4. 最大误差点保留正确的 case、time、node 和坐标。
5. 预测 HDF5 schema、完成标记、checkpoint SHA256 和复用判断正确。
6. 三角形 PyG `face` 可正确导出 VTU，PVD 时间与帧文件一致。
7. 使用一个真实测试工况的单步前向检查两个模型输出形状、绝对值还原和持久化流程。

## 9. 非目标

本阶段不生成论文排版图，不评价训练集或验证集，不重新训练模型，也不将不同工况的相同 node index 解释为同一物理空间点。跨工况比较通过聚合指标完成；单节点时序只在同一工况内跟踪，因为该工况网格拓扑在时间上不变。
