# Transolver 黏滞阻尼器代理模型

详细文档：

- `技术文档.md`：数据、动网格、共享归一化、Physics-Attention、训练目标和公平对比；
- `使用说明.md`：环境、训练、断点恢复、TensorBoard、rollout、结果读取和故障排查。

本目录提供从 COMSOL HDF5 数据到训练、验证、checkpoint、断点恢复和整段时间序列递推的完整 Transolver 流程。核心 Physics-Attention 来自 THUML 官方 Transolver irregular-mesh 实现，数据口径则直接复用 `meshGraphNet_self`，用于公平比较两个模型。

## 1. 代码来源与目录

官方仓库已拉取到：

```text
transolver_self/official_source/Transolver
```

核对版本为 `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。官方源码目录带有独立 `.git`，因此被主项目忽略；实际训练使用 `transolver_self/model/` 内的适配实现，不依赖 `timm` 或 `einops`。MIT 许可保存在 `THIRD_PARTY_LICENSE.txt`。

主要文件：

```text
transolver_self/
├── model/physics_attention.py  # 官方不规则网格 Physics-Attention
├── model/transolver.py         # Transolver 主干
├── model/simulator.py          # PyG、共享特征和归一化适配
├── train.py                    # 训练、验证、日志、存档和恢复
├── rollout.py                  # 自回归时间序列预测并还原动态网格
└── test_pipeline.py            # 变节点 batch、前向、反向和推理测试
```

## 2. 数据复用关系

数据仍由 `meshGraphNet_self.dataset.FpcDataset` 读取，因此两种模型使用完全相同的：

- HDF5 静态参考网格、三角形拓扑、`p` 和 `T` 时间序列；
- JSON 几何参数、加载参数和材料参数；
- 三段网格运动公式，以及任意时刻实时恢复的 `pos` 和 `mesh_velocity`；
- 按工况文件排序后的 80%/10%/10% train/valid/test 划分；
- PyG `Data` 和 `DataLoader` batch；
- 一步增量目标、训练损失、验证指标和 checkpoint 基础格式。

注意：COMSOL 尚未全部算完时，HDF5 文件数会继续增长，80%/10%/10% 的边界也会变化。正式对比实验应在全部工况完成后开始，或固定一份数据文件清单。

## 3. 单个图对象

加载器返回当前时刻 `t` 到下一时刻 `t+1` 的训练样本：

| 属性 | 形状 | 含义 |
|---|---:|---|
| `x[:, 1:3]` | `[N,2]` | 当前 `p_t, T_t` |
| `y` | `[N,2]` | 下一时刻 `p_(t+1), T_(t+1)` |
| `pos` | `[N,2]` | 根据时间实时恢复的节点坐标 |
| `face` | `[3,F]` | 静态三角形拓扑 |
| `mesh_velocity` | `[N,2]` | 当前节点网格速度 |
| `mesh_region` | `[N]` | 下段、中段、上段区域编号 |
| `case_features` | `[1,10]` | 几何、加载和材料参数 |
| `time` | `[1]` | 当前时间 |
| `piston_displacement` | `[1]` | 活塞位移 |
| `piston_velocity` | `[1]` | 活塞速度 |

`face` 会随结果保留，用于恢复图网格和后处理。Transolver 本身不做边消息传递，因此不会把 `face/edge_index` 输入注意力层。

## 4. 特征与归一化

共享实现位于 `meshGraphNet_self/features.py`。PyG batch 先以扁平节点形式累计统计量，再按图打包为带 mask 的稠密张量。即使未来不同工况节点数不同，padding 节点也不会参与 Physics-Attention。

Transolver 输入为：

```text
position: [B,N,2]
  = normalized(pos_R, pos_Z)

function: [B,N,20]
  = normalized(p_t, T_t)                         2
  + normalized(mesh_velocity_R, mesh_velocity_Z) 2
  + one_hot(mesh_region)                         3
  + normalized(case/time/motion context)        13
```

其中 13 维 context 为：

```text
c, sx, sy, r1, a2, b1, b2, A, Ts, mu,
time, piston_displacement, piston_velocity
```

`field_normalizer`、`position_normalizer`、`mesh_velocity_normalizer`、`context_normalizer` 和 `output_normalizer` 均直接使用 `meshGraphNet_self.utils.normalization.Normalizer`。统计量只在训练模式累计，并随模型 `state_dict` 写入 checkpoint。

训练目标和 MeshGraphNet 相同：

```text
target_delta = [p_(t+1)-p_t, T_(t+1)-T_t]
loss = MSE(predicted_normalized_delta, normalized_target_delta)
```

## 5. 模型结构

默认配置：

| 参数 | 默认值 |
|---|---:|
| hidden size | 256 |
| Physics-Attention blocks | 8 |
| attention heads | 8 |
| physics slices | 32 |
| MLP ratio | 1 |
| dropout | 0 |
| output | 2 (`Δp, ΔT`) |

流程为：节点特征映射到 256 维，按每个 head 学习节点到 32 个 physics slice 的权重，在 slice token 之间做自注意力，再投影回原节点。前 7 层输出隐藏状态，最后一层输出每个节点的两个归一化增量。

默认 Transolver 约 2,817,346 个参数；当前默认 MeshGraphNet 约 2,879,234 个参数，两者规模接近，适合直接比较。改变 `hidden-size/layers/heads/slice-num` 后应重新记录参数量。

## 6. 开始训练

在项目根目录执行：

```powershell
conda activate pinn
python transolver_self\train.py
```

常用显式参数：

```powershell
python transolver_self\train.py `
  --data-root "计算有限元数据\comsol_hdf5" `
  --epochs 100 `
  --batch-size 4 `
  --hidden-size 256 `
  --layers 8 `
  --heads 8 `
  --slice-num 32
```

训练复用 MeshGraphNet 的 `train_one_epoch` 和 `evaluate`，所以损失与评价口径完全一致。每个 epoch 输出：

- `train_mse`：归一化增量训练 MSE；
- `valid_mse`：归一化增量验证 MSE；
- `p_rmse`：下一时刻压力物理量 RMSE；
- `T_rmse`：下一时刻温度物理量 RMSE。

输出文件：

```text
transolver_self/checkpoints/last.pt
transolver_self/checkpoints/best.pt
transolver_self/checkpoints/epoch_XXXX.pt
transolver_self/checkpoints/training_config.json
transolver_self/runs/...
```

断点恢复：

```powershell
python transolver_self\train.py `
  --resume transolver_self\checkpoints\last.pt
```

## 7. 整段时间序列预测

训练完成后，从某个真实初始场开始自回归预测。每一步都会根据新时间重新计算节点位置、网格速度、活塞位移和活塞速度：

```powershell
python transolver_self\rollout.py `
  --checkpoint transolver_self\checkpoints\best.pt `
  --case-id Case_0300 `
  --steps 150
```

默认输出到 `transolver_self/rollouts/Case_0300_rollout.pt`。读取方式：

```python
import torch

result = torch.load(
    "transolver_self/rollouts/Case_0300_rollout.pt",
    weights_only=False,
)
mesh_t = result["meshes"][20]

mesh_t.pos               # 当前时刻动态节点坐标 [N,2]
mesh_t.face              # 静态拓扑 [3,F]
mesh_t.predicted_fields  # 预测的 p、T [N,2]
mesh_t.p                 # 压力 [N,1]
mesh_t.T                 # 温度 [N,1]
mesh_t.mesh_velocity     # 当前网格速度 [N,2]
mesh_t.time              # 当前时间
result["metrics"]        # 整段 rollout 的 p/T RMSE
```

## 8. 验证

使用指定的 `pinn` 环境：

```powershell
D:\Aanconda3\envs\pinn\python.exe -m unittest transolver_self.test_pipeline -v
```

该测试覆盖不同节点数图组成 batch、padding mask、模型前向、反向传播和下一步物理量预测。真实数据冒烟测试还应确认实际 HDF5 样本能在 GPU 上完成一次前向与反向。
