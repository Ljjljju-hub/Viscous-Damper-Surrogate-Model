import pyvista as pv
import h5py
import numpy as np
import re
from pathlib import Path
import traceback


def vtu_to_hdf5(vtu_filepath: Path, h5_filepath: Path):
    """核心转换逻辑（支持动态捕获移动网格的所有场变量）"""
    print(f"📥 正在读取: {vtu_filepath.name}")
    mesh = pv.read(str(vtu_filepath))

    array_names = mesh.point_data.keys()

    # 【核心修复 1】放宽正则表达式：捕获任何变量名 (包括 u, v, x, y, 或带下划线的变量)
    pattern = re.compile(r"(.+)_@_t=(.*)")

    time_steps = set()
    # 【核心修复 2】不再写死 p 和 T，改为动态字典，自适应 VTU 里的所有变量
    fields_dict = {}

    for name in array_names:
        match = pattern.match(name)
        if match:
            var_name = match.group(1).strip()
            t_val = float(match.group(2))
            time_steps.add(t_val)

            # 如果是第一次遇到这个变量，在字典里为它开辟空间
            if var_name not in fields_dict:
                fields_dict[var_name] = {}
            fields_dict[var_name][t_val] = mesh.point_data[name]

    if not time_steps:
        raise ValueError(
            f"⚠️ 文件 {vtu_filepath.name} 中没有解析到任何有效的时间步数据 (可能是空文件)。"
        )

    sorted_times = sorted(list(time_steps))
    num_steps = len(sorted_times)
    num_points = mesh.number_of_points

    # ================= 开始写入 HDF5 =================
    with h5py.File(str(h5_filepath), 'w') as f_h5:
        # 1. 静态网格域 (初始参考坐标)
        mesh_group = f_h5.create_group("mesh")
        mesh_group.create_dataset("coordinates", data=mesh.points, compression="gzip")
        mesh_group.create_dataset("connectivity", data=mesh.cells, compression="gzip")

        # 2. 时间轴
        f_h5.create_dataset(
            "time_steps", data=np.array(sorted_times), compression="gzip"
        )

        # 3. 动态场域
        fields_group = f_h5.create_group("fields")

        # 【核心修复 3】动态遍历所有提取到的场（不仅是 p 和 T，连同移动网格变量一起存入！）
        for var_name in fields_dict.keys():
            if not fields_dict[var_name]:
                continue

            var_matrix = np.zeros((num_steps, num_points), dtype=np.float32)
            for i, t in enumerate(sorted_times):
                if t in fields_dict[var_name]:
                    var_matrix[i, :] = fields_dict[var_name][t]

            fields_group.create_dataset(var_name, data=var_matrix, compression="gzip")

    # 打印出提取到了哪些场，方便你在终端直接验证网格变量是否被成功抓取
    extracted_vars = list(fields_dict.keys())
    print(f"✅ 成功生成: {h5_filepath.name} ({num_points}节点, {num_steps}步)")
    print(f"   -> 已提取变量: {extracted_vars}")


def batch_convert_dir(input_dir, output_dir):
    """
    批量扫描并转换文件夹内的所有 VTU 文件。
    """
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()

    if not in_path.exists() or not in_path.is_dir():
        print(f"❌ 错误: 输入文件夹不存在 -> {in_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录准备就绪: {out_path}")

    vtu_files = list(in_path.glob("*.vtu"))
    total_files = len(vtu_files)

    if total_files == 0:
        print(f"⚠️ 在 {in_path} 中没有找到任何 .vtu 文件。")
        return

    print(f"🔍 共找到 {total_files} 个 VTU 文件，开始批量转换...\n" + "-" * 40)

    success_count = 0
    for idx, vtu_file in enumerate(vtu_files, 1):
        h5_file = out_path / f"{vtu_file.stem}.h5"

        print(f"[{idx}/{total_files}] 处理中...")
        try:
            vtu_to_hdf5(vtu_file, h5_file)
            success_count += 1
        except Exception as e:
            print(f"❌ 转换 {vtu_file.name} 时失败: {e}")
            # traceback.print_exc() # 去掉注释可以查看具体的报错堆栈
        print("-" * 40)

    print(f"\n🎉 批量处理完成！成功: {success_count}/{total_files}")
    print(f"📂 HDF5 文件已保存至: {out_path}")


if __name__ == "__main__":
    # ================= 核心路径修改 =================
    # 获取当前 Python 脚本所在的绝对目录
    BASE_DIR = Path(__file__).parent.resolve()

    # 以脚本所在目录为基准，拼接子文件夹
    INPUT_FOLDER = BASE_DIR / "comsol_output"  # 指向脚本同级目录下的 VTU 数据
    OUTPUT_FOLDER = BASE_DIR / "comsol_hdf5"  # 指向脚本同级目录下的 HDF5 存放处
    # ===============================================

    batch_convert_dir(INPUT_FOLDER, OUTPUT_FOLDER)
