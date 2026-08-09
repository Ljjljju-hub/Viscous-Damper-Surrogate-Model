# 域外泛化数据集设计

## 1. 目标

在现有 1000 个训练域工况之外生成一套独立的 COMSOL 数据集，用于评价 MeshGraphNet 和 Transolver 对未见参数范围的外推能力。

本阶段负责：

1. 生成并校验域外参数 JSON；
2. 分批运行 COMSOL，并在批次之间彻底退出 worker、JVM 和 server；
3. 将 VTU 原子转换为 HDF5；
4. 保存断点状态、失败记录、日志和参数审计结果。

神经网络推理和域外误差对比不属于本阶段。后续评价入口直接读取本工作区生成的 HDF5 和参数 JSON。

## 2. 训练域与域外范围

原训练参数范围如下：

| 参数族 | 参数 | 训练范围 |
|---|---|---:|
| geometry | `c` | `[1, 3] mm` |
| geometry | `sx` | `[40, 120] mm` |
| geometry | `sy` | `[120, 320] mm` |
| geometry | `r1` | `[50, 70] mm` |
| geometry | `a2` | `[40, 80] mm` |
| geometry | `b1` | `[80, 120] mm` |
| geometry | `b2` | `[80, 160] mm` |
| loading | `A` | `[10, 90] mm` |
| loading | `Ts` | `[0.1, 0.5] s` |
| material | `mu` | `[1000, 3000] Pa*s` |

“域外 10%”定义为：在训练区间两侧各增加一个宽度为训练区间宽度 10% 的外推带。域外带不包含训练边界本身。

| 参数 | 下侧域外带 | 上侧域外带 |
|---|---:|---:|
| `c` | `[0.8, 1)` | `(3, 3.2]` |
| `sx` | `[32, 40)` | `(120, 128]` |
| `sy` | `[100, 120)` | `(320, 340]` |
| `r1` | `[48, 50)` | `(70, 72]` |
| `a2` | `[36, 40)` | `(80, 84]` |
| `b1` | `[76, 80)` | `(120, 124]` |
| `b2` | `[72, 80)` | `(160, 168]` |
| `A` | `[2, 10)` | `(90, 98]` |
| `Ts` | `[0.06, 0.1)` | `(0.5, 0.54]` |
| `mu` | `[800, 1000)` | `(3000, 3200]` |

## 3. 分组采样

总计生成 150 个工况，固定随机种子，保证可复现：

| 分组 | 数量 | 域外规则 |
|---|---:|---|
| `geometry_ood` | 50 | 7 个几何参数中恰好一个位于域外带 |
| `loading_ood` | 50 | `A`、`Ts` 中恰好一个位于域外带 |
| `material_ood` | 50 | `mu` 位于域外带 |

每个样本只有目标参数族可以越界，并且目标参数族中恰好一个参数越界。其他参数继续在原训练域内使用 Latin Hypercube Sampling。目标参数、上侧越界和下侧越界尽量均衡分配；数量不能整除时，各类别数量最多相差 1。

工况编号使用 `Case_1001` 到 `Case_1150`。结果存放在独立目录，因此不会覆盖原数据；同时该命名继续兼容当前 Dataset、失败检测和文件扫描规则。

所有样本必须满足：

```text
sy >= b2 + 2*A + 10 mm
```

生成器通过拒绝采样处理不满足安全余量的组合。取整后再次执行严格校验，防止数值落回训练边界。

## 4. JSON 与审计信息

主参数文件继续命名为 `4_Combined_Master_Dataset.json`，保留现有字段：

```text
case_id
geometry
loading
material
```

每个样本增加 `ood` 字段：

```text
group
parameter
side
training_lower
training_upper
value
normalized_ood_distance
```

其中：

```text
normalized_ood_distance = 到最近训练边界的距离 / 该参数训练区间宽度
```

该值位于 `(0, 0.1]`，可用于后续绘制“外推距离-预测误差”曲线。

同时生成：

- `parameter_audit.csv`：每个工况一行，包含全部参数、域外标签和几何安全余量；
- `dataset_summary.json`：记录随机种子、分组数量、参数上下侧数量和校验结果。

## 5. 工作区结构

根目录新增：

```text
ood_generalization_workspace/
├── generate_parameters.py
├── run_pipeline.py
├── README.md
├── 4_Combined_Master_Dataset.json
├── parameter_audit.csv
├── dataset_summary.json
├── comsol_output/          # VTU，Git 忽略
├── comsol_hdf5/            # HDF5，Git 忽略
├── batch_logs/             # 运行日志，Git 忽略
├── batch_state.json        # 断点状态，Git 忽略
└── failed_cases.json       # 终态失败记录，Git 忽略
```

参数生成文件和审计文件提交到 Git；大型求解结果和运行状态不提交。

## 6. COMSOL 流程复用

不复制 `计算有限元数据` 中的 worker、控制器和 HDF5 转换器。现有入口增加可选的工作区参数：

```text
--workspace-root
--model-path
```

默认值保持现状，因此原 1000 个工况的计算行为不改变。域外入口将新工作区路径传给控制器，控制器继续复用：

1. HDF5/VTU 完整性扫描；
2. 已失败工况默认跳过；
3. 每批独立 Python、MPh、JVM 和 COMSOL server；
4. worker 完全退出后才转换本批 HDF5；
5. 临时文件和原子替换；
6. 中断后重启续算。

`transfer2hdf5.py` 已支持输入和输出目录参数，控制器只需显式传入域外 VTU/HDF5 路径。

## 7. 统一入口

`run_pipeline.py` 文件末尾集中放置用户参数：

```python
GENERATE_PARAMETERS = True
OVERWRITE_PARAMETERS = False
DRY_RUN = True
BATCH_SIZE = 10
CORES = 16
MAX_RETRIES = 2
PAUSE_SECONDS = 10.0
TIMEOUT_MINUTES = 0.0
SHOW_WORKER_WINDOW = True
```

执行顺序为：

```text
生成或读取参数 -> 全量校验 -> 调用 COMSOL 控制器 -> 分批求解 -> 转换 HDF5 -> 汇总状态
```

默认 `DRY_RUN=True`，第一次运行只检查参数和显示批次。用户确认计划后改为 `False`，再次运行同一个 Python 文件即可开始自动计算。

## 8. 错误处理

- 参数 JSON 已存在且 `OVERWRITE_PARAMETERS=False` 时不重新抽样，只执行校验；
- 参数文件不满足 150 个唯一工况、分组数量或域外规则时拒绝启动 COMSOL；
- 单个工况明确求解失败后写入独立 `failed_cases.json`，后续默认跳过；
- VTU 存在但 HDF5 不存在时只补转换；
- HDF5 已完整时直接跳过；
- 原训练数据目录和域外目录之间不得互相扫描、覆盖或共享状态文件。

## 9. 验证

自动化测试至少覆盖：

1. 固定种子重复生成得到相同参数；
2. 三组均为 50 个工况；
3. 每个工况恰好一个指定参数越界；
4. 非目标参数均在训练域内；
5. 上下侧和目标参数分配均衡；
6. 所有几何安全余量不小于 10 mm；
7. JSON 可由现有数据读取器解析；
8. 域外 dry-run 使用新目录，不读取原训练结果；
9. 未指定新参数时，原 COMSOL 入口仍使用原目录。
