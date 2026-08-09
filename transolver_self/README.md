# Transolver 黏滞阻尼器代理模型

本目录保留 THUML Transolver 的项目适配、训练 worker、rollout 和许可信息。数据加载、动网格、特征、归一化、训练目标与评价指标复用 `meshGraphNet_self`，以保证两个模型公平比较。

```text
transolver_self/
├── model/
│   ├── physics_attention.py # 不规则网格 Physics-Attention
│   ├── transolver.py        # Transolver 主干
│   └── simulator.py         # PyG 和共享特征适配
├── train_worker.py          # 调度器使用的 Transolver 内部 worker
├── rollout.py               # 自回归时间序列预测
├── official_source/README.md
├── THIRD_PARTY_LICENSE.txt
└── 技术文档.md
```

官方实现版本固定为：

```text
75e0f67643806a81cd1d3f6adc88dd8c02416fe7
```

## 训练入口

不要直接运行本目录的 `train_worker.py`。统一修改 [training_workspace/train.py](../training_workspace/train.py) 末尾的变量区，然后运行：

```powershell
conda activate pinn
python training_workspace\train.py
```

只训练 Transolver 时设置：

```python
MODELS = ["transolver"]
TRAIN_SIZES = [100, 200, 400, 800]
SEEDS = [42]
DRY_RUN = False
```

训练结果保存在：

```text
training_workspace/runs/transolver/
└── n0100/seed_42/
    ├── checkpoints/best.pt
    ├── checkpoints/last.pt
    ├── tensorboard/
    ├── metrics.csv
    ├── summary.json
    └── train.log
```

Transolver 默认使用 8 层、256 隐藏维、8 个 attention head 和 32 个 physics slice。详细结构与复用关系见 [技术文档.md](技术文档.md)。
