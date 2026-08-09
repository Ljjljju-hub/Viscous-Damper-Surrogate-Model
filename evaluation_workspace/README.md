# 模型测试与时序可视化

本目录有两个独立人工入口：

```text
test.py                  完整 test split 单步预测与分层指标
visualize_timeseries.py  指定工况时序推理与 ParaView 导出
```

都使用 `pinn` 环境，不需要输入命令行参数。运行前只修改对应文件末尾的变量区。

## 1. 完整测试

```powershell
conda activate pinn
python evaluation_workspace\test.py
```

默认读取：

```text
training_workspace/runs/meshgraphnet/n0100/seed_42/checkpoints/best.pt
training_workspace/runs/transolver/n0100/seed_42/checkpoints/best.pt
training_workspace/dataset_split/split_manifest.json
```

`test.py` 对 test split 中每个 HDF5 的 150 个相邻时间步执行单步预测。每一步输入真实当前场，输出绝对下一时刻 `p/T`，不是自回归 rollout。

预测保存在：

```text
results/test/n0100_seed42/predictions/<模型>/Case_XXXX.h5
```

每个文件包含目标时刻、动态位置、网格速度、静态拓扑、区域、真实场、绝对预测场、checkpoint 哈希及输出归一化统计量。文件完整且 checkpoint 哈希未变化时，下一次运行会跳过推理并直接复用。

## 2. 分层指标

结果目录包含：

| 文件 | 分析层级 |
|---|---|
| `summary.csv` | 每个模型汇总全部 test 工况、时间步和节点 |
| `case_metrics.csv` | 每个模型、每个 HDF5 工况 |
| `time_metrics.csv` | 每个模型、统一时间索引，跨全部工况 |
| `case_time_metrics.csv` | 每个模型、单个 HDF5、单个时间步 |
| `extrema.csv` | 全局及每工况最大误差点的位置和数值 |
| `percentiles.csv` | P95、P99 和最大绝对误差 |
| `model_comparison.png` | 两个模型总体指标对比 |

总体 RMSE 先汇总所有节点时刻的平方误差，除以总数量后只开一次根号，不对各工况 RMSE 做简单平均。

相对误差只在：

```text
abs(truth) >= 1% * 该工况该字段的 max(abs(truth))
```

的节点时刻计算。CSV 同时记录有效点、排除点和实际阈值。RMSE、MAE、P95、P99 和最大绝对误差仍使用所有节点。

## 3. 时序可视化

先在 `visualize_timeseries.py` 末尾设置：

```python
CASE_ID = "Case_0866"
START_INDEX = 0
STEPS = None
SOURCE_MODE = "saved_one_step"
CASE_SELECTION_MODE = "manual"
```

然后运行：

```powershell
python evaluation_workspace\visualize_timeseries.py
```

模式：

- `saved_one_step`：读取 `test.py` 保存的预测，不加载模型。
- `rollout`：第 0 帧使用真实初始场，后续持续使用前一步模型预测。

`CASE_SELECTION_MODE`：

- `manual`：只导出手工设置的 `CASE_ID`。
- `test_extremes`：读取完整测试的 `case_metrics.csv`，自动选择每个模型、每个字段的 RMSE 最大/最小、工况最坏单点绝对误差最大/最小、工况最坏有效相对误差最大/最小。

自动模式会生成 `representative_cases.csv`。相同 Case 只导出一次 PVD，但 CSV 保留它被选中的所有原因。每个代表工况目录还会保存：

```text
ground_truth/Case_XXXX.h5   原始 COMSOL GT
selection_reasons.csv       当前 Case 的所有选择原因
comparison.pvd              组合时序入口
frames/*.vtu                全部时间帧
step_metrics.csv
error_vs_time.png
```

这里“单点误差最小”不是在所有节点里寻找接近零的误差，而是比较各工况的最坏单点误差，选择其中最小者。

输出目录：

```text
results/visualization/<Case>/start_<索引>/<模式>/
├── comparison.pvd
├── frames/frame_XXXX.vtu
├── step_metrics.csv
└── error_vs_time.png
```

用 ParaView 打开 `comparison.pvd`。每个节点可选择真实 `p/T`、两个模型预测、绝对误差、有效相对误差、区域和网格速度。PVD 依赖 `frames` 文件夹，移动结果时需要一起保留。

## 4. 建议分析顺序

1. 用 `summary.csv` 判断两个模型总体泛化能力。
2. 用 `case_metrics.csv` 找出困难几何和加载工况。
3. 用 `time_metrics.csv` 检查误差是否集中于特定运动阶段。
4. 用 `case_time_metrics.csv` 定位具体工况和时刻。
5. 用 `extrema.csv` 定位最差节点，再通过 PVD 查看周围误差云图。
6. 自回归结果重点查看 `error_vs_time.png`，判断误差是否随预测步数累积。
