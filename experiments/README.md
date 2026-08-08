# 数据规模对比实验

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

训练规模使用 `100,200,...,800`。验证和测试工况永远不变，因此曲线只反映训练数据规模变化。详细统计见 [dataset_scale/README.md](dataset_scale/README.md)。

## 2. 生成固定 split

COMSOL 全部工况进入成功或失败终态后执行：

```powershell
conda activate pinn

python experiments\create_split_manifest.py --force
```

输出：

```text
experiments/dataset_scale/split_manifest.json
experiments/dataset_scale/case_split.json
experiments/dataset_scale/case_index.csv
```

manifest 包含：

- 固定打乱后的 `train_pool`、`valid` 和 `test` case ID；
- 39 个终态失败 case ID 和未归类 case 检查结果；
- 每个 HDF5 的文件大小、修改时间、节点数和时间步数；
- JSON 参数文件 SHA256；
- 无效或尚未写完的 HDF5 列表。

脚本默认要求参数 JSON 中的 1000 个工况全部属于有效 HDF5 或终态失败。存在未归类工况时会拒绝生成正式 manifest。临时测试可以增加 `--allow-incomplete`，但不能用于最终论文实验。

## 3. 先检查将执行哪些任务

```powershell
& '.\experiments\开始训练.bat' --dry-run
```

默认实验矩阵：

```text
models = meshgraphnet, transolver
train sizes = 100,200,...,800
seeds = 42,43,44
```

共 `2 × 8 × 3 = 48` 次训练。

## 4. 开始全部训练

训练结果独立保存在：

```text
experiments/dataset_scale/runs/
├── meshgraphnet/
│   ├── n0100/seed_42/
│   └── ...
└── transolver/
    ├── n0100/seed_42/
    └── ...
```

模型、数据规模和随机种子共同确定一个独立实验目录，两个模型的 checkpoint、日志和指标不会混合。

推荐先检查单个正式任务：

```powershell
& '.\experiments\开始训练.bat' --dry-run `
  --models meshgraphnet `
  --train-sizes 100 `
  --seeds 42
```

正式启动全部 48 个任务：

```powershell
& '.\experiments\开始训练.bat' `
  --epochs 100 `
  --batch-size 4 `
  --early-stopping-patience 15 `
  --device auto
```

只运行部分规模：

```powershell
& '.\experiments\开始训练.bat' `
  --models meshgraphnet transolver `
  --train-sizes 100 200 400 800 `
  --seeds 42 43 44
```

单次快速验证必须写入独立的 `_smoke` 目录，不能污染正式 `runs/`：

```powershell
& '.\experiments\开始训练.bat' `
  --output-root experiments/_smoke/training `
  --models transolver `
  --train-sizes 100 `
  --seeds 42 `
  --epochs 2
```

RTX 4060 上建议先用 `--batch-size 2` 验证显存，再提高到 4。用于公平对比的 MeshGraphNet 与 Transolver 正式实验应保持相同 batch size。

## 5. 断电恢复

每个实验保存在独立目录：

```text
experiments/dataset_scale/runs/
└── transolver/
    └── n0100/
        └── seed_42/
            ├── checkpoints/
            │   ├── last.pt
            │   ├── best.pt
            │   └── training_config.json
            ├── tensorboard/
            ├── metrics.csv
            ├── summary.json
            ├── status.json
            └── train.log
```

断电后重新运行完全相同的 `run_scale_study.py` 命令：

- 有 `summary.json`：任务已经完成，自动跳过；
- 有 `last.pt` 但没有 summary：从最近完整 epoch 自动恢复；
- 什么都没有：从头训练。

checkpoint 保存模型、optimizer、scheduler、Normalizer、early stopping 计数，以及 Python、NumPy、CPU/CUDA、DataLoader generator RNG 状态。恢复发生在 epoch 边界；断电时正在运行的当前 epoch 仍需重算。

调度器会保存模型、规模、seed、epoch、batch size、学习率和 split manifest 指纹。若同一结果目录已经完成，但新命令配置不同，程序会报错而不会静默跳过；需要做新配置实验时使用新的 `--output-root`。

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

## 7. rollout 评价

所有训练完成后，对固定 test 集前 10 个工况运行完整自回归序列：

```powershell
python experiments\evaluate_scale_study.py `
  --rollout-count 10 `
  --device auto
```

每完成一个 test case 就更新 `evaluation.json`。断电后重新执行同一命令，已经完成的 case 会跳过。

默认只保存指标，避免 48 个实验产生大量网格序列文件。如果确实需要每一帧 PyG 网格：

```powershell
python experiments\evaluate_scale_study.py `
  --rollout-count 10 `
  --save-predictions
```

MeshGraphNet rollout 每步会根据新 `pos` 重新生成动态边特征；Transolver 每步重新生成动态位置、网格速度和稠密节点输入。

## 8. 汇总和绘图

```powershell
python experiments\plot_scale_study.py
```

输出：

```text
experiments/dataset_scale/plots/
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
