import json
import numpy as np
from scipy.stats import qmc
from pathlib import Path

def generate_decoupled_lhs_datasets(target_samples=1000, output_dir="./Damper_Datasets"):
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 数据集保存路径: {out_path}")

    # 1. 定义上下界
    bounds_geo = {
        'c': [1.0, 3.0], 'sx': [40, 120], 'sy': [120, 320],
        'r1': [50, 70], 'a2': [40, 80], 'b1': [80, 120], 'b2': [80, 160]
    }
    bounds_load = {
        'A': [10, 90], 'Ts': [0.1, 0.5]
    }
    bounds_mat = {
        'mu': [1000.0, 3000.0] 
    }

    valid_geo, valid_load, valid_mat, valid_combined = [], [], [], []
    
    sampler_geo = qmc.LatinHypercube(d=len(bounds_geo))
    sampler_load = qmc.LatinHypercube(d=len(bounds_load))
    sampler_mat = qmc.LatinHypercube(d=len(bounds_mat))

    print(f"⏳ 正在抽样并执行几何过滤，目标样本: {target_samples} 个...")

    # 2. 循环抽样与组合过滤
    while len(valid_combined) < target_samples:
        batch_size = 500
        
        sample_geo = qmc.scale(sampler_geo.random(n=batch_size), 
                               [v[0] for v in bounds_geo.values()], [v[1] for v in bounds_geo.values()])
        sample_load = qmc.scale(sampler_load.random(n=batch_size), 
                                [v[0] for v in bounds_load.values()], [v[1] for v in bounds_load.values()])
        sample_mat = qmc.scale(sampler_mat.random(n=batch_size), 
                               [v[0] for v in bounds_mat.values()], [v[1] for v in bounds_mat.values()])

        for i in range(batch_size):
            g = dict(zip(bounds_geo.keys(), np.round(sample_geo[i], 2)))
            l = dict(zip(bounds_load.keys(), np.round(sample_load[i], 3)))
            m = dict(zip(bounds_mat.keys(), np.round(sample_mat[i], 2)))

            # 核心约束条件：腔室高度 >= 活塞高度 + 双倍振幅 + 安全余量(10mm)
            if g['sy'] >= (g['b2'] + 2 * l['A'] + 10.0):
                
                # 为每个子集生成带有 Part 说明的专属 ID
                base_index = f"{len(valid_combined) + 1:04d}"
                case_id = f"Case_{base_index}"
                
                g['part_id'] = f"Geo_Sample_{base_index}"
                l['part_id'] = f"Load_Sample_{base_index}"
                m['part_id'] = f"Mat_Sample_{base_index}"

                valid_geo.append(g)
                valid_load.append(l)
                valid_mat.append(m)
                
                # 汇总表包含统一的 Case ID，并嵌套三个带有专属 ID 的字典
                valid_combined.append({
                    "case_id": case_id,
                    "geometry": g,
                    "loading": l,
                    "material": m
                })

            if len(valid_combined) == target_samples:
                break

    # 3. 辅助函数保存 JSON
    def save_json(filename, description, data_list):
        filepath = out_path / filename
        json_structure = {
            "dataset_metadata": {
                "description": description,
                "total_samples": len(data_list),
                "sampling_method": "Latin Hypercube Sampling (LHS) with geometric safety filtering"
            },
            "parameters_list": data_list
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=4, ensure_ascii=False)

    # 4. 生成 4 个 JSON 文件
    save_json("1_Geometry_Parameters.json", "阻尼器核心几何尺寸 (mm)。已过滤干涉废案。", valid_geo)
    save_json("2_Loading_Parameters.json", "阻尼器动态加载策略：振幅 A(mm) 与周期 Ts(s)。", valid_load)
    save_json("3_Material_Parameters.json", "流体介质 25℃ 基础动力粘度 mu (Pa·s)。", valid_mat)
    save_json("4_Combined_Master_Dataset.json", "主数据集。合并几何、加载、材料，用于 COMSOL 遍历。", valid_combined)
    
    print("✅ 4个 JSON 文件全部生成完毕！")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    target_folder = script_dir / "Damper_Parameters_Datasets"
    generate_decoupled_lhs_datasets(target_samples=1000, output_dir=target_folder)