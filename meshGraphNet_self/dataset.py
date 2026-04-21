import h5py
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data, Dataset
from utils.utils import NodeType


def apply_mesh_movement(initial_pos, t, amplitude, freq):
    """
    根据时间步 t 动态计算网格的最新坐标。

    参数:
        initial_pos (np.array): 初始参考坐标 [N, 2]
        t (float): 当前物理时间 (秒)
        amplitude (float): 振幅 (A)
        freq (float): 频率 (f, Hz)

    返回:
        np.array: 变形后的最新坐标 [N, 2]
    """
    # 复制一份原始坐标，防止修改到 Dataset 内部的缓存
    current_pos = initial_pos.copy()

    # 核心公式: x = A * sin(2 * pi * f * t)
    # 计算当前的位移量
    omega = 2 * np.pi * freq
    displacement = amplitude * np.sin(omega * t)

    # 📍 逻辑分支：根据你的仿真需求选择位移施加方式

    # 模式 1：整体平移 (如果你的整个网格都跟着活塞动)
    # current_pos[:, 0] += displacement  # 假设沿 X 轴移动

    # 模式 2：局部移动 (只有特定区域动，比如 X > 0.05 的区域)
    # 这种方式最符合阻尼器的真实物理逻辑
    move_mask = initial_pos[:, 0] > 0.05  # 找到活塞端的所有节点
    current_pos[move_mask, 0] += displacement

    return current_pos


class FpcDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train'):
        super().__init__(data_root, transform=None, pre_transform=None)

        self.data_root = Path(data_root).resolve()
        self.split = split

        all_files = sorted(list(self.data_root.glob("*.h5")))
        if not all_files:
            raise FileNotFoundError(
                f"❌ 在 {self.data_root} 目录下没有找到任何 .h5 文件！"
            )

        # 计算 8:1:1 的切分索引
        num_files = len(all_files)
        train_idx = int(num_files * 0.8)
        valid_idx = int(num_files * 0.9)

        # 根据 split 参数分配文件
        if split == 'train':
            self.files = all_files[:train_idx]  # 前 80%
        elif split == 'valid':
            self.files = all_files[train_idx:valid_idx]  # 80% 到 90%
        elif split == 'test':
            self.files = all_files[valid_idx:]  # 最后 10%
        else:
            self.files = all_files  # 默认或 'all'，加载全部

        self.index_map = []
        for fpath in self.files:
            with h5py.File(str(fpath), 'r') as f:
                num_steps = len(f['time_steps'][:])
                for t in range(num_steps - 1):
                    self.index_map.append((fpath, t))

        print(f"[{split.upper()}] 数据集加载完毕: 共计 {len(self.index_map)} 张图。")

    def len(self):
        return len(self.index_map)

    def get(self, idx):
        file_path, t_idx = self.index_map[idx]

        with h5py.File(str(file_path), 'r') as f:
            # 1. 提取当前步骤的真实物理时间 t (用于代入你的位移公式)
            t_val = f['time_steps'][t_idx]

            # 2. 获取网格拓扑
            cells = f['mesh/connectivity'][:]
            faces = cells.reshape(-1, 4)[:, 1:]
            face = torch.tensor(faces.T, dtype=torch.long)

            # ========================================================
            # 📍 核心区：在这里修改移动的坐标
            # ========================================================
            # 获取原始的静态参考坐标
            pos_numpy = f['mesh/coordinates'][:, :2].copy()

            # 👇 将下面的公式替换为你 COMSOL 里真实的移动函数 👇
            # 假设你的网格是沿 X 轴做正弦运动 (例如振幅 A=0.01, 频率 f=1.0)
            A = 0.01
            omega = 2 * np.pi * 1.0
            displacement_x = A * np.sin(omega * t_val)

            # 如果是整体平移，直接加到所有节点的 X 坐标上
            # （如果你只有部分区域移动，你可以通过判断 pos_numpy 的范围来局部施加）
            pos_numpy[:, 0] += displacement_x

            # 将更新后的坐标转化为 Tensor
            pos = torch.tensor(pos_numpy, dtype=torch.float32)
            # ========================================================

            # 3. 获取物理场特征 (p, T)
            p_t = torch.tensor(f['fields/p'][t_idx, :], dtype=torch.float32).unsqueeze(
                1
            )
            T_t = torch.tensor(f['fields/T'][t_idx, :], dtype=torch.float32).unsqueeze(
                1
            )

            p_next = torch.tensor(
                f['fields/p'][t_idx + 1, :], dtype=torch.float32
            ).unsqueeze(1)
            T_next = torch.tensor(
                f['fields/T'][t_idx + 1, :], dtype=torch.float32
            ).unsqueeze(1)

        # 4. 组装张量
        N = pos.shape[0]
        node_type = torch.full((N, 1), NodeType.NORMAL, dtype=torch.float32)

        x = torch.cat([node_type, p_t, T_t], dim=-1)
        y = torch.cat([p_next, T_next], dim=-1)

        # 此时返回的 pos 已经是根据时间 t 动态计算出变形位置的真实坐标了
        return Data(x=x, y=y, pos=pos, face=face)
