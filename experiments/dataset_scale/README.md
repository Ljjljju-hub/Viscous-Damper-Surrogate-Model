# 最终数据统计与递增训练划分

## 1. 最终工况统计

统计时间：2026-08-08。参数文件共包含 1000 个工况，所有工况均已有明确终态。

| 状态 | 数量 | 占比 |
|---|---:|---:|
| 有效 HDF5 | 961 | 96.1% |
| COMSOL 终态失败 | 39 | 3.9% |
| 尚未计算或状态不明 | 0 | 0% |

961 个 HDF5 均通过以下结构校验：

- 包含 `mesh/coordinates`、`mesh/connectivity` 和 `time_steps`；
- 包含训练目标 `fields/p` 和 `fields/T`；
- 每个文件均为 725 个节点、151 个时间步；
- 字段形状均与节点数、时间步数一致；
- 文件总大小约 0.608 GiB，单文件大小为 609790 到 733391 字节。

终态失败工况：

```text
Case_0091, Case_0124, Case_0130, Case_0147, Case_0159,
Case_0165, Case_0184, Case_0339, Case_0340, Case_0370,
Case_0376, Case_0394, Case_0412, Case_0426, Case_0433,
Case_0440, Case_0441, Case_0448, Case_0481, Case_0497,
Case_0517, Case_0614, Case_0618, Case_0629, Case_0640,
Case_0712, Case_0726, Case_0754, Case_0766, Case_0774,
Case_0800, Case_0863, Case_0864, Case_0872, Case_0890,
Case_0896, Case_0919, Case_0946, Case_0968
```

## 2. 固定数据划分

有效工况使用 Python `random.Random(42)` 做一次固定打乱，然后划分为：

| 数据组 | 工况数 | 用途 |
|---|---:|---|
| `train_pool` | 800 | 构造递增训练集 |
| `valid` | 80 | 早停和最佳 checkpoint 选择 |
| `test` | 81 | 最终单步及 rollout 评价 |

三个集合无重复、无交集，合计恰好覆盖全部 961 个有效工况。39 个失败工况不会进入任何训练或评价集合。

## 3. 递增训练集

训练规模固定为：

```text
100, 200, 300, 400, 500, 600, 700, 800
```

规模为 `N` 时始终使用：

```python
train_case_ids = train_pool[:N]
```

因此这些子集严格嵌套：100 个工况集合包含于 200 个工况集合，依次类推。数据规模曲线不会因为每个规模重新随机抽样而混入额外的样本差异。

## 4. 清单文件

- `case_split.json`：可移植的固定 train/valid/test case ID、失败编号和标准训练规模，纳入 Git。
- `case_index.csv`：1000 个参数 case 的逐行状态、split、训练顺序和 HDF5 结构统计，纳入 Git。
- `split_manifest.json`：训练程序使用的本机完整快照，额外保存绝对路径、文件大小和修改时间；该文件不纳入 Git。

`case_index.csv` 中，训练集工况的 `train_order` 为 1 到 800。某个规模 `N` 使用所有 `train_order <= N` 的行。

## 5. 重新生成与验证

在仓库根目录运行：

```powershell
conda activate pinn
python experiments\create_split_manifest.py --force
& '.\experiments\开始训练.bat' --dry-run
```

生成器要求 1000 个参数工况全部属于“有效 HDF5”或“终态失败”。只要存在未归类 case，正式清单生成就会失败。

开始完整规模实验：

```powershell
& '.\experiments\开始训练.bat' `
  --epochs 100 `
  --batch-size 4 `
  --early-stopping-patience 15 `
  --device auto
```
