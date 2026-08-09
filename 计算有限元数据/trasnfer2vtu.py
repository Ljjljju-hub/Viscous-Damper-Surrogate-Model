import pyvista as pv
import h5py
import numpy as np
from pathlib import Path


def export_timeseries_to_pvd(h5_filepath: Path, output_dir: Path):
    """
    将包含所有时间步的 HDF5 文件导出为 ParaView 可播放的 PVD 动画序列，
    并支持动态移动网格 (ALE) 的实时坐标刷新。
    """
    # 1. 创建专门存放动画帧的文件夹 (parents=True 连带创建父目录)
    output_dir.mkdir(parents=True, exist_ok=True)
    pvd_filepath = output_dir / "animation.pvd"

    with h5py.File(str(h5_filepath), 'r') as f:
        print(f"📖 正在打开 HDF5 数据集: {h5_filepath.name}")

        # ================= 1. 重建基础网格拓扑 (只做一次) =================
        points = f['mesh/coordinates'][:]
        cells = f['mesh/connectivity'][:]

        pts_per_cell = cells[0]
        num_cells = len(cells) // (pts_per_cell + 1)

        if pts_per_cell == 3:
            cell_type = np.full(num_cells, pv.CellType.TRIANGLE, dtype=np.uint8)
        elif pts_per_cell == 4:
            # 注意：如果是 2D 四边形，请改为 pv.CellType.QUAD
            cell_type = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
        elif pts_per_cell == 8:
            cell_type = np.full(num_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        else:
            raise ValueError(f"未知的单元节点数: {pts_per_cell}")

        # 实例化基础网格
        grid = pv.UnstructuredGrid(cells, cell_type, points)
        print(
            f"📐 几何重建完毕: {grid.number_of_points} 节点, {grid.number_of_cells} 单元"
        )

        time_steps = f['time_steps'][:]
        num_steps = len(time_steps)

        # ================= 2. 准备 PVD 播放列表的 XML 头部 =================
        pvd_content = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>',
        ]

        print(f"🎞️ 开始导出 {num_steps} 帧动画数据 (包含动态网格更新)...")

        # ================= 3. 循环注入物理场并导出每一帧 =================
        for i, t_val in enumerate(time_steps):

            # 3.1 把当前时刻 (t) 的所有标量场 (p, T 等) 挂载到网格上
            for var_name in f['fields'].keys():
                grid.point_data[var_name] = f[f'fields/{var_name}'][i, :]

            # 3.2 动态更新网格几何坐标 (核心修复点)
            # 方案 A：如果 HDF5 中存了实时的 x, y 坐标场
            if 'x' in f['fields'] and 'y' in f['fields']:
                new_x = f['fields/x'][i, :]
                new_y = f['fields/y'][i, :]
                # 兼容 2D/3D：如果有 z 则读取，否则补 0
                new_z = (
                    f['fields/z'][i, :] if 'z' in f['fields'] else np.zeros_like(new_x)
                )
                # 覆盖网格点的真实物理坐标
                grid.points = np.column_stack((new_x, new_y, new_z))

            # 方案 B：如果 HDF5 内存的是 u, v 位移场
            elif 'u' in f['fields'] and 'v' in f['fields']:
                initial_points = f['mesh/coordinates'][:]
                disp_x = f['fields/u'][i, :]
                disp_y = f['fields/v'][i, :]
                disp_z = (
                    f['fields/w'][i, :] if 'w' in f['fields'] else np.zeros_like(disp_x)
                )
                # 初始坐标 + 位移
                grid.points = initial_points + np.column_stack((disp_x, disp_y, disp_z))

            # 3.3 保存当前帧为独立的 VTU 文件 (例如: frame_0000.vtu)
            vtu_filename = f"frame_{i:04d}.vtu"
            vtu_filepath = output_dir / vtu_filename
            grid.save(str(vtu_filepath))

            # 3.4 在播放列表中记录这一帧的真实物理时间
            pvd_content.append(
                f'    <DataSet timestep="{t_val}" group="" part="0" file="{vtu_filename}"/>'
            )

            # 打印进度条
            if i % 10 == 0 or i == num_steps - 1:
                print(
                    f"   -> 已导出 {i+1}/{num_steps} 帧 (真实物理时间 t={t_val:.4f}s)"
                )

        # ================= 4. 闭合并保存 PVD 播放列表 =================
        pvd_content.append('  </Collection>')
        pvd_content.append('</VTKFile>')

        with open(pvd_filepath, 'w') as pvd_file:
            pvd_file.write("\n".join(pvd_content))

        print("\n✅ 动画全序列导出成功！")
        print(f"👉 终极操作指南：")
        print(f"   1. 打开 ParaView。")
        print(f"   2. 找到并打开文件： {pvd_filepath}")
        print(
            f"   3. 点击 ParaView 顶部的 'Play (▶️)' 按钮，即可观看网格运动与物理场动画。"
        )


if __name__ == "__main__":
    # ================= 核心路径修改 =================
    # 锁定当前 Python 脚本所在的绝对物理目录
    BASE_DIR = Path(__file__).parent.resolve()

    # 指向上一节生成的 HDF5 数据文件夹 (comsol_hdf5) 里的特定文件
    H5_FILE = BASE_DIR / "comsol_hdf5" / "Case_0050.h5"

    # 动画输出文件夹也锁定在脚本同级目录下的 paraview_animation 文件夹中
    OUTPUT_DIR = BASE_DIR / "paraview_animation"
    # ===============================================

    if not H5_FILE.exists():
        print(f"❌ 找不到 HDF5 文件: {H5_FILE}")
        print(
            "请检查：\n1. 上一步是否成功生成了该文件\n2. 文件名 (Case_0050.h5) 是否与实际一致。"
        )
    else:
        export_timeseries_to_pvd(H5_FILE, OUTPUT_DIR)
