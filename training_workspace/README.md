# 统一训练工作目录

本目录自动完成以下流程：

```text
冻结 HDF5 数据快照
-> 固定 train pool / valid / test
-> 按 100、200、... 构造嵌套训练子集
-> MeshGraphNet 与 Transolver 多随机种子训练
-> 自动断点恢复和跳过已完成任务
-> 固定测试集单步评价
-> 固定测试工况自回归 rollout
-> 汇总均值、标准差并输出 PNG/PDF 曲线
```

## 目录结构

```text
training_workspace/
├── train.py                    # 唯一人工训练入口，参数定义在文件末尾
├── dataset_split/              # 固定的数据集划分和本机数据快照
│   ├── case_split.json
│   ├── case_index.csv
│   └── split_manifest.json
├── runs/
│   ├── meshgraphnet/           # MeshGraphNet 按规模和 seed 保存
│   └── transolver/             # Transolver 按规模和 seed 保存
├── plots/                      # 汇总 CSV 及 PNG/PDF 曲线
├── run_scale_study.py          # 可恢复训练调度器
├── create_split_manifest.py    # 固定数据划分生成器
├── evaluate_scale_study.py     # 批量 rollout 评价
└── plot_scale_study.py         # 指标汇总和绘图
```

训练结果自动形成 `runs/<模型>/n<规模>/seed_<种子>/`，无需人工新建实验目录。

## 1. 最终数据和划分

1000 个参数工况的最终计算结果为：

```text
有效 HDF5       = 961
COMSOL 终态失败 = 39
未归类           = 0
```

有效数据使用种子 42 固定划分：

```text
train pool = 800
valid      = 80
test       = 81
```

训练规模使用 `100,200,...,800`。验证和测试工况永远不变，因此曲线只反映训练数据规模变化。详细统计见 [dataset_split/README.md](dataset_split/README.md)。

## 2. 生成固定 split

COMSOL 全部工况进入成功或失败终态后执行：

```powershell
conda activate pinn

python training_workspace\create_split_manifest.py --force
```

输出：

```text
training_workspace/dataset_split/split_manifest.json
training_workspace/dataset_split/case_split.json
training_workspace/dataset_split/case_index.csv
```

manifest 包含：

- 固定打乱后的 `train_pool`、`valid` 和 `test` case ID；
- 39 个终态失败 case ID 和未归类 case 检查结果；
- 每个 HDF5 的文件大小、修改时间、节点数和时间步数；
- JSON 参数文件 SHA256；
- 无效或尚未写完的 HDF5 列表。

脚本默认要求参数 JSON 中的 1000 个工况全部属于有效 HDF5 或终态失败。存在未归类工况时会拒绝生成正式 manifest。临时测试可以增加 `--allow-incomplete`，但不能用于最终论文实验。

## 3. Python 训练入口

唯一人工训练入口是本目录的 `train.py`。不输入命令行参数，只修改文件末尾变量区：

```python
MODELS = ["meshgraphnet", "transolver"]
TRAIN_SIZES = [100]
SEEDS = [42]

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1.0e-4
EARLY_STOPPING_PATIENCE = 15
SAVE_EVERY = 10
BATCH_LOG_EVERY = 10
NUM_WORKERS = 0
DEVICE = "auto"

MESHGRAPHNET_HIDDEN_SIZE = 128
MESSAGE_PASSING_STEPS = 15

TRANSOLVER_HIDDEN_SIZE = 256
TRANSOLVER_LAYERS = 8
TRANSOLVER_HEADS = 8
TRANSOLVER_SLICE_NUM = 32
TRANSOLVER_DROPOUT = 0.0
TRANSOLVER_MLP_RATIO = 1

DRY_RUN = True
CONTINUE_ON_ERROR = False
OUTPUT_ROOT = WORKSPACE_ROOT / "runs"
```

变量含义：

- `MODELS`：可选 `meshgraphnet`、`transolver` 或两者；
- `TRAIN_SIZES`：从 `100,200,...,800` 中任意选择；设为 `None` 时读取 manifest 中全部规模；
- `SEEDS`：一个或多个随机种子；
- `MESHGRAPHNET_*`：MeshGraphNet 隐藏维度和消息传递层数；
- `TRANSOLVER_*`：Transolver 隐藏维度、层数、head、slice 和 dropout；
- `DRY_RUN=True`：只显示计划，不训练；确认后改为 `False`；
- `CONTINUE_ON_ERROR`：单个任务失败后是否继续后续组合；正式首次运行建议保持 `False`；
- `OUTPUT_ROOT`：模型、checkpoint、日志和指标的总输出目录。

运行方式固定为：

```powershell
conda activate pinn
python training_workspace\train.py
```

## 4. 模型和规模选择

训练结果独立保存在：

```text
training_workspace/runs/
├── meshgraphnet/
│   ├── n0100/seed_42/
│   └── ...
└── transolver/
    ├── n0100/seed_42/
    └── ...
```

模型、数据规模和随机种子共同确定一个独立实验目录，两个模型的 checkpoint、日志和指标不会混合。

只训练 MeshGraphNet 的 100 工况：

```python
MODELS = ["meshgraphnet"]
TRAIN_SIZES = [100]
SEEDS = [42]
```

比较两个模型的部分规模：

```python
MODELS = ["meshgraphnet", "transolver"]
TRAIN_SIZES = [100, 200, 400, 800]
SEEDS = [42]
```

完整 48 个任务：

```python
MODELS = ["meshgraphnet", "transolver"]
TRAIN_SIZES = [100, 200, 300, 400, 500, 600, 700, 800]
SEEDS = [42, 43, 44]
```

短训练验证必须写入独立的 `_smoke` 目录，不能污染正式结果：

```python
MODELS = ["transolver"]
TRAIN_SIZES = [100]
SEEDS = [42]
EPOCHS = 2
OUTPUT_ROOT = WORKSPACE_ROOT / "_smoke"
```

RTX 4060 8 GB 的真实前向、反向和 Adam 更新测试表明，`BATCH_SIZE = 16`
时 MeshGraphNet 峰值约 4.48 GiB、Transolver 约 1.27 GiB。正式公平对比时
两个模型统一使用 16；若同时运行其他占用显存的程序，可降为 8。

## 5. 断电恢复

每个实验保存在独立目录：

```text
training_workspace/runs/
└── transolver/
    └── n0100/
        └── seed_42/
            ├── checkpoints/
            │   ├── last.pt
            │   ├── best.pt
            │   ├── normalization_stats.pt
            │   └── training_config.json
            ├── tensorboard/
            ├── metrics.csv
            ├── summary.json
            ├── status.json
            └── train.log
```

断电后保持 `training_workspace/train.py` 末尾变量不变，再次运行 `python training_workspace\train.py`：

- 有 `summary.json`：任务已经完成，自动跳过；
- 有 `last.pt` 但没有 summary：从最近完整 epoch 自动恢复；
- 什么都没有：从头训练。

checkpoint 保存模型、optimizer、scheduler、Normalizer、early stopping 计数，以及 Python、NumPy、CPU/CUDA、DataLoader generator RNG 状态。恢复发生在 epoch 边界；断电时正在运行的当前 epoch 仍需重算。

新实验在第一个训练 epoch 前额外执行一次 `fit normalization`：只扫描训练工况，按
`工况×150 个相邻时间步×节点` 统计当前场、动态坐标、网格速度、上下文和目标增量。
统计使用 `float64` batch-Welford 算法，完成后冻结并保存到
`checkpoints/normalization_stats.pt`。后续所有 epoch、验证集和测试集只读取该统计，
不会继续累计，也不会使用验证/测试信息。

调度器会保存模型、规模、seed、epoch、batch size、学习率和 split manifest 指纹。若同一结果目录已经完成，但变量配置不同，程序会报错而不会静默跳过；新配置实验应修改 `OUTPUT_ROOT`。

## 6. 指标和曲线保存

`metrics.csv` 每个 epoch 一行：

```text
epoch
global_step
train_normalized_mse
valid_normalized_mse
valid_rmse_p
valid_rmse_T
learning_rate
epoch_seconds
```

CSV 采用按 epoch 覆盖的原子更新，恢复训练不会制造重复行。

`summary.json` 在训练和固定 test 集评价完成后生成：

```text
best epoch
best valid normalized MSE
test normalized MSE
test p RMSE
test T RMSE
参数量
训练工况和图样本数量
累计 epoch 时间
```

TensorBoard 曲线保存在每个任务自己的 `tensorboard` 目录，不会与其他规模混合。
训练过程中每 `BATCH_LOG_EVERY` 个 batch 实时写入
`loss/train_batch_normalized_mse`；每个完整 epoch 结束后写入训练集、验证集、
物理量 RMSE、学习率和 epoch 耗时。TensorBoard 默认每 10 秒刷新事件文件。

## 7. rollout 评价

所有训练完成后，对固定 test 集前 10 个工况运行完整自回归序列：

```powershell
python training_workspace\evaluate_scale_study.py `
  --rollout-count 10 `
  --device auto
```

每完成一个 test case 就更新 `evaluation.json`。断电后重新执行同一命令，已经完成的 case 会跳过。

默认只保存指标，避免 48 个实验产生大量网格序列文件。如果确实需要每一帧 PyG 网格：

```powershell
python training_workspace\evaluate_scale_study.py `
  --rollout-count 10 `
  --save-predictions
```

MeshGraphNet rollout 每步会根据新 `pos` 重新生成动态边特征；Transolver 每步重新生成动态位置、网格速度和稠密节点输入。

## 8. 汇总和绘图

```powershell
python training_workspace\plot_scale_study.py
```

输出：

```text
training_workspace/plots/
├── individual_results.csv
├── aggregate.csv
├── best_valid_normalized_mse.png/.pdf
├── test_normalized_mse.png/.pdf
├── test_rmse_p.png/.pdf
├── test_rmse_T.png/.pdf
├── rollout_rmse_p.png/.pdf
├── rollout_rmse_T.png/.pdf
├── total_epoch_seconds.png/.pdf
├── training_curves_meshgraphnet.png/.pdf
└── training_curves_transolver.png/.pdf
```

规模曲线显示 3 个随机种子的均值，并用误差棒表示样本标准差。

## 9. 公平比较约束

自动流程保证：

- 两个模型读取同一个 manifest；
- 每个规模使用同一个嵌套训练 case 子集；
- valid/test 永久固定；
- 相同规模和 seed 使用独立但同种子的 DataLoader generator；
- 相同 batch size、学习率、最大 epoch 和 early stopping；
- 都按 valid normalized MSE 选择 `best.pt`；
- 都在同一个 test 集计算单步指标；
- 都在同一批 test case 计算 rollout 指标。

模型特有部分仍保持各自设计：MeshGraphNet 使用动态网格边，Transolver 使用 Physics-Attention。
