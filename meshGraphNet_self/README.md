# Viscous Damper MeshGraphNet

本目录只保留 MeshGraphNet 运行所需代码和技术说明。

```text
meshGraphNet_self/
├── dataset.py          # HDF5、JSON 和单步图样本读取
├── mesh_motion.py      # 三段动网格位置与速度恢复
├── features.py         # 两模型共享特征
├── graph.py            # face -> edge_index / edge_attr
├── training.py         # 两模型共享训练、评价和 checkpoint
├── experiment_utils.py # 固定划分、RNG 恢复和指标文件工具
├── train_worker.py     # 调度器使用的 MeshGraphNet 内部 worker
├── model/              # Encoder-Processor-Decoder 网络
├── utils/              # NodeType 和 Normalizer
└── 技术文档.md
```

## 训练入口

不要直接运行本目录的 `train_worker.py`。统一修改 [training_workspace/train.py](../training_workspace/train.py) 末尾的变量区，然后运行：

```powershell
conda activate pinn
python training_workspace\train.py
```

只训练 MeshGraphNet 时设置：

```python
MODELS = ["meshgraphnet"]
TRAIN_SIZES = [100, 200, 400, 800]
SEEDS = [42]
DRY_RUN = False
```

训练结果保存在：

```text
training_workspace/runs/meshgraphnet/
└── n0100/seed_42/
    ├── checkpoints/best.pt
    ├── checkpoints/last.pt
    ├── tensorboard/
    ├── metrics.csv
    ├── summary.json
    └── train.log
```

重新运行相同的 `training_workspace/train.py` 会自动跳过已完成任务，或从 `last.pt` 恢复中断任务。详细设计见 [技术文档.md](技术文档.md)，完整实验流程见 [training_workspace/README.md](../training_workspace/README.md)。
