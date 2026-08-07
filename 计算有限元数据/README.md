# COMSOL 有限元数据自动续算

## 文档导航

- [自动续算使用说明.md](自动续算使用说明.md)：面向运行人员，包含启动、续算、参数、日志和故障处理。
- [自动续算技术说明.md](自动续算技术说明.md)：面向开发维护，说明进程隔离、状态判定、原子写入和重试实现。

## 1. 推荐运行方式

后续统一使用 conda `pinn` 环境。先仅查看待计算计划，不启动 COMSOL：

```powershell
conda activate pinn
python .\计算有限元数据\run_remaining.py --dry-run
```

确认计划后启动完整流程：

```powershell
python .\计算有限元数据\run_remaining.py --batch-size 10 --cores 16 --max-retries 2 --pause-seconds 10
```

也可以使用固定指向 `pinn` 环境的入口：

```powershell
& '.\计算有限元数据\运行剩余工况.bat' --batch-size 10 --cores 16
```

不要使用一个长期运行的 `main.py` 计算全部工况。`main.py` 现在是单批 worker，由 `run_remaining.py` 自动调用。

## 2. 每批执行流程

控制器执行以下循环：

1. 读取 `4_Combined_Master_Dataset.json` 中全部 `case_id`。
2. 校验已有 HDF5 和 VTU，只选择真正缺失的工况。
3. 为当前批次启动新的 `pinn` Python 进程和独立 Windows 终端。
4. worker 启动新的 MPh client-server、加载母版、逐个求解并原子导出 VTU。
5. 一批结束后卸载模型、断开 client、停止 server、退出 worker；独立终端随之关闭。
6. 控制器等待 worker 完全退出，再用独立转换进程把本批 VTU 原子转换为 HDF5。
7. 等待 `pause-seconds`，然后启动下一批全新进程。
8. 完整扫描剩余项，失败工况最多重试 `max-retries` 次。

这保证 Python、JVM 和 COMSOL server 都不会跨批次复用。MPh 本身在同一 Python 进程中只允许一个 client，因此这里有意使用独立进程作为内存释放边界。

## 3. 断点续算与文件安全

- 有效 HDF5 视为已完成，不重复计算。
- 没有 HDF5 但存在完整 VTU 时，只补做 HDF5 转换。
- VTU 与 HDF5 都不存在或结构无效时，重新运行 COMSOL。
- VTU 先写入 `.partial.vtu`，成功后原子替换正式文件。
- HDF5 同样先写临时文件，通过结构校验后才原子替换。
- 按 `Ctrl+C` 中断后，当前 worker 进程树会被关闭；再次运行同一命令即可续算。

运行状态写入 `batch_state.json`，详细日志写入 `batch_logs/`。这两个运行产物已加入 `.gitignore`。

## 4. 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--batch-size` | 10 | 每次 COMSOL 进程计算的工况数；内存仍高时降为 5 或 2 |
| `--cores` | 16 | 单个 COMSOL 会话使用的核心数 |
| `--max-retries` | 2 | 首次失败后额外重试次数 |
| `--pause-seconds` | 10 | 两批之间的等待时间 |
| `--timeout-minutes` | 0 | 单批超时，0 表示不限时；超时会终止整个 worker 进程树 |
| `--first-case` | 1 | 只处理指定起始序号，包含该序号 |
| `--last-case` | 1000 | 只处理指定结束序号，包含该序号 |
| `--dry-run` | 否 | 只打印分批计划，不计算、不转换 |
| `--no-worker-window` | 否 | 不显示每批独立终端，后台运行 |
| `--no-convert` | 否 | 只生成 VTU，不自动生成 HDF5 |

例如只续算第 305 到 500 个工况：

```powershell
python .\计算有限元数据\run_remaining.py --first-case 305 --last-case 500 --batch-size 5
```

如果某批长时间无响应，可设置单批超时：

```powershell
python .\计算有限元数据\run_remaining.py --batch-size 5 --timeout-minutes 180
```

## 5. 单独运行与测试

通常不需要手工调用 worker。调试单个工况时可以运行：

```powershell
python .\计算有限元数据\main.py --case-ids Case_0305 --cores 16
```

运行不启动 COMSOL 的自动化测试：

```powershell
python -m unittest discover -s .\计算有限元数据 -p "test_*.py" -v
```

## 6. HDF5 文件结构

hdf5文件：详细的结构
┣ 📂 mesh # 静态网格域（整个文件仅存一次）
┃ ┣ 📜 coordinates # 节点几何坐标
┃ ┗ 📜 connectivity # 单元拓扑连接关系
┣ 📜 time_steps # 瞬态求解的时间轴记录
┗ 📂 fields # 动态物理场域（按变量解耦）
┣ 📜 p # 全时域压力场矩阵
┗ 📜 T # 全时域温度场矩阵。

1. 📂 mesh: 静态网格基底 (整个训练只读一次)这部分数据与时间无关。在 PyTorch 中，你只需要在 Dataset 的 __init__ 或第一次 get() 时读取即可。📜 coordinates (节点几何坐标)矩阵形状: [N, 3]数据类型: float32 或 float64内存长相:Plaintext[
  [0.0,  0.0,  0.0],  # 节点 0 的 x, y, z
  [0.01, 0.0,  0.0],  # 节点 1 的 x, y, z
  ...
]
如何读取: pos = torch.tensor(f['mesh/coordinates'][:])。如果是 2D 仿真，你可以直接切片 [:, :2] 扔掉 Z 轴。📜 connectivity (单元拓扑连接关系)矩阵形状: [E * (pts_per_cell + 1)] （这是一个一维数组）数据类型: int32 或 int64内存长相 (以 2D 三角形为例):VTK/PyVista 存网格时，会在每个单元前面加一个“节点数”标识（三角形是 3，四边形是 4）。Plaintext[ 3, 0, 1, 2,   3, 1, 3, 2,   3, ... ]
 👆 单元1的节点 👆 单元2的节点
避坑指南: 你不能直接把它当成 edge_index！你需要把它重塑为二维矩阵，并砍掉第一列的那个 3：Python# 假设是三角形网格
cells = f['mesh/connectivity'][:]
faces = cells.reshape(-1, 4)[:, 1:] # 变成 [E, 3] 的矩阵

2. 📜 time_steps: 物理时间轴
矩阵形状: [S] （一维数组）数据类型: float32
内存长相: [0.0, 0.0036, 0.0072, 0.0108, ..., 0.54]
用途: 这个数组主要是给人看的，或者用于在 PINNs（物理信息神经网络）中作为求导的 $dt$。在标准的 MeshGraphNet 自回归训练中，你其实可以直接靠索引（index）去取数据，不用读具体的时间值。

3. 📂 fields: 动态场数据 (极其高频的切片区)这是 HDF5 结构设计的真正精髓所在。 所有的物理场都被严格规定为 [时间步, 节点数] 的二维矩阵。
📜 p 和 📜 T
矩阵形状: [S, N]数据类型: float32
内存长相:
Plaintext         节点0   节点1   节点2  ... 节点N
t=0   [ 1e5,    1e5,    1e5,   ... ]  <-- 第 0 步的全域压力
t=1   [ 1.1e5,  1.2e5,  1e5,   ... ]  <-- 第 1 步的全域压力
...
t=150 [ 2e5,    1.5e5,  9e4,   ... ]
为什么要把 $S$ 放在第一维？ (C-Contiguous 内存连续性)硬盘读取数据是一段一段连续读的。当你在 DataLoader 里想拿“第 10 步的全部压力”时，代码写成 f['fields/p'][10, :]。因为第 10 步的所有节点数据在硬盘上是挨在一起的，HDF5 会瞬间把这块连续内存拍给显卡，耗时几乎为 0。
