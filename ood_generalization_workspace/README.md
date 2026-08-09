# 域外泛化数据集工作区

该目录用于生成和计算独立于原 1000 个训练工况的域外数据。它复用 `计算有限元数据` 中已经验证过的 COMSOL 分批计算、进程退出、断点续算和 HDF5 转换逻辑，但参数、结果、日志和失败记录都保存在本目录，不会扫描或覆盖原训练数据。

## 1. 数据集组成

固定生成 150 个工况：

| 分组 | 工况数 | 规则 |
|---|---:|---|
| `geometry_ood` | 50 | 7 个几何参数中恰好一个超出训练范围 |
| `loading_ood` | 50 | `A` 或 `Ts` 中恰好一个超出训练范围 |
| `material_ood` | 50 | `mu` 超出训练范围 |

域外带宽等于对应训练区间宽度的 10%，上下两侧尽量均衡。每个样本只有一个参数越界，其他参数仍从原训练范围进行 LHS 采样，并满足：

```text
sy >= b2 + 2*A + 10 mm
```

工况编号为 `Case_1001` 到 `Case_1150`。具体参数范围见 [设计文档](../docs/superpowers/specs/2026-08-09-ood-generalization-dataset-design.md)。

## 2. 文件结构

```text
ood_generalization_workspace/
├── generate_parameters.py              # 只生成参数文件
├── parameter_generator.py              # 采样、校验和审计实现
├── run_pipeline.py                     # 自动计算统一入口
├── 4_Combined_Master_Dataset.json       # COMSOL 使用的域外参数
├── parameter_audit.csv                  # 每个工况及其域外标签
├── dataset_summary.json                 # 分组统计和校验摘要
├── comsol_output/                       # COMSOL VTU，运行后生成
├── comsol_hdf5/                         # 转换后的 HDF5，运行后生成
├── batch_logs/                          # controller/worker 日志
├── batch_state.json                     # 当前断点状态
└── failed_cases.json                    # 明确失败的终态工况
```

## 3. 第一次运行

后续统一使用 conda `pinn` 环境：

```powershell
conda activate pinn
python .\ood_generalization_workspace\run_pipeline.py
```

`run_pipeline.py` 默认设置：

```python
DRY_RUN = True
```

因此第一次运行只会完成以下操作：

1. 读取并严格校验 150 个参数；
2. 检查 OOD 工作区已有的 VTU/HDF5；
3. 显示 15 个批次，每批 10 个工况；
4. 不启动 COMSOL，不产生求解结果。

日志中的路径必须指向 `ood_generalization_workspace`，待计算数应为 150，不能出现原数据集的完成数量。

## 4. 开始自动计算

确认 dry-run 计划后，只修改 `run_pipeline.py` 末尾：

```python
DRY_RUN = False
```

然后再次执行同一个入口：

```powershell
python .\ood_generalization_workspace\run_pipeline.py
```

脚本会自动执行：

```text
参数校验
-> 启动一批独立 Python/COMSOL 会话
-> 求解并原子发布 VTU
-> 完全退出 worker、JVM 和 COMSOL server
-> 转换本批 HDF5
-> 等待后启动下一批
```

不需要 `.bat` 文件，也不需要手动逐批关闭终端。

## 5. 运行参数

所有常用参数都位于 `run_pipeline.py` 末尾：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `GENERATE_PARAMETERS` | `True` | 参数缺失时自动生成；已有参数只校验、不重抽样 |
| `OVERWRITE_PARAMETERS` | `False` | 是否主动替换已有 JSON/CSV/summary |
| `DRY_RUN` | `True` | 只显示计划，不启动 COMSOL |
| `BATCH_SIZE` | `10` | 每个独立 COMSOL 进程计算的工况数 |
| `CORES` | `16` | 单个 COMSOL 会话使用的核心数 |
| `MAX_RETRIES` | `2` | 非明确工况失败时的额外批次尝试次数 |
| `PAUSE_SECONDS` | `10.0` | 两批之间等待时间 |
| `TIMEOUT_MINUTES` | `0.0` | 单批超时；`0` 表示不限制 |
| `SHOW_WORKER_WINDOW` | `True` | 是否显示每批独立终端 |

如果仍然出现明显的内存增长，优先把 `BATCH_SIZE` 从 10 降到 5 或 2。

## 6. 断点续算

计算中断后直接再次运行 `run_pipeline.py`：

- 有效 HDF5：直接跳过；
- 只有完整 VTU：只补 HDF5 转换；
- VTU/HDF5 都没有：重新计算；
- 已写入 `failed_cases.json`：默认跳过，避免无限重算。

原训练目录中的 `Case_0001` 到 `Case_1000` 不参与任何完成状态判断。

## 7. 参数文件安全

域外参数使用固定随机种子 `20260809`。一旦开始 COMSOL 计算，不要修改随机种子、范围、首个编号，也不要把 `OVERWRITE_PARAMETERS` 改为 `True`，否则同一文件名可能对应不同物理参数。

需要重新设计另一套域外数据时，应复制成新的独立工作区和编号，而不是覆盖已经求解的数据。

`parameter_audit.csv` 中保存：

- 10 个输入参数；
- 域外分组、目标参数和上下侧；
- 相对于训练区间宽度的域外距离；
- 几何安全余量。

这些列可以直接用于后续按 `geometry_ood`、`loading_ood`、`material_ood` 分组统计神经网络误差。

## 8. 单独重新生成参数

仅在尚未开始 COMSOL 计算时使用：

```powershell
python .\ood_generalization_workspace\generate_parameters.py
```

参数文件已存在时，该脚本默认拒绝覆盖。确需替换时先确认本目录没有关联 VTU/HDF5，再把 `generate_parameters.py` 中的 `OVERWRITE` 改为 `True`。
