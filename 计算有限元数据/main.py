# 基于Mph库，使用标准comsol模型，批量计算不同参数下的结果
import json
import time
import mph
import traceback
from pathlib import Path

def run_comsol_batch(start_case=1, end_case=None, max_samples=None):
    """
    运行 COMSOL 批处理。

    参数:
        start_case (int): 起始案例编号（1-based，例如 51 对应 Case_0051），默认为 1。
        end_case   (int|None): 结束案例编号（不包含，1-based），例如 end_case=200 会运行到 Case_0199。
                               若为 None，则依据 max_samples 决定结束位置。
        max_samples (int|None): 最大运行案例数量（从 start_case 开始连续运行）。
                                若 end_case 不为 None，则忽略 max_samples。
    """
    # ================= 1. 动态相对路径配置 =================
    base_dir = Path(__file__).parent.resolve()
    json_path = base_dir / "4_Combined_Master_Dataset.json"
    model_path = base_dir / "standard_model.mph"
    output_dir = base_dir / "comsol_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = base_dir / "failed_samples_log.txt"

    # ================= 2. 读取参数集 =================
    print(f"📖 正在加载数据集: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        samples = dataset["parameters_list"]
        total_samples = len(samples)
    except FileNotFoundError:
        print(f"❌ 找不到 JSON 文件，请检查路径: {json_path}")
        return

    # 确定实际要运行的索引范围 (0-based)
    start_idx = start_case - 1
    if start_idx < 0 or start_idx >= total_samples:
        print(f"❌ 起始案例编号 {start_case} 超出有效范围 (1 ~ {total_samples})")
        return

    if end_case is not None:
        end_idx = end_case - 1
        if end_idx > total_samples:
            end_idx = total_samples
        if end_idx <= start_idx:
            print(f"⚠️ 结束案例编号 {end_case} 不大于起始编号 {start_case}，没有案例需要运行。")
            return
    else:
        if max_samples is not None:
            end_idx = min(start_idx + max_samples, total_samples)
        else:
            end_idx = total_samples

    run_count = end_idx - start_idx
    start_case_actual = start_idx + 1
    end_case_actual = end_idx  # 注意：最后一个运行的案例编号是 end_idx（因为 end_idx 是索引，+1 才是编号）
    print(f"🎯 运行案例编号: {start_case_actual} 到 {end_case_actual} (共 {run_count} 个，总样本数 {total_samples})")

    # ================= 3. 启动 COMSOL 引擎 =================
    def start_comsol_client():
        """启动 COMSOL 客户端并确保连接可用"""
        print("🚀 正在启动 COMSOL 客户端...")
        client = mph.start(cores=16)
        # 等待连接就绪（尝试加载一个临时空模型验证）
        for attempt in range(3):
            try:
                test_model = client.load(str(model_path))
                test_model.clear()
                print("✅ COMSOL 客户端连接正常")
                return client
            except Exception as e:
                print(f"⚠️ 连接测试失败 (尝试 {attempt+1}/3): {e}")
                time.sleep(2)
        raise RuntimeError("无法连接到 COMSOL 服务器")

    client = start_comsol_client()

    # ================= 4. 批处理主循环 =================
    for idx, sample in enumerate(samples[start_idx:end_idx]):
        global_index = start_idx + idx
        case_id = sample["case_id"]
        vtu_filename = output_dir / f"{case_id}.vtu"

        # 断点续传
        if vtu_filename.exists():
            print(f"⏩ {case_id} 已存在，跳过计算。 ({idx+1}/{run_count})")
            continue

        print(f"\n▶️ 开始计算 {case_id} ({idx+1}/{run_count})...")
        start_time = time.time()

        # 【内存防爆机制】每跑完 50 个样本，强制重启一次客户端
        if idx > 0 and idx % 50 == 0:
            # print("🧹 触发内存清理：正在重启 COMSOL 客户端...")
            # try:
            #     client.disconnect()
            # except:
            #     pass
            # time.sleep(10)
            # client = start_comsol_client()
            pass

        # 加载干净的母版模型（带重试）
        model = None
        for load_attempt in range(2):
            try:
                model = client.load(str(model_path))
                break
            except Exception as e:
                print(f"⚠️ 加载模型失败 (尝试 {load_attempt+1}/2): {e}")
                if load_attempt == 0:
                    print("🔄 尝试重启 COMSOL 客户端...")
                    try:
                        client.disconnect()
                    except:
                        pass
                    time.sleep(2)
                    client = start_comsol_client()
                else:
                    raise
        if model is None:
            raise RuntimeError("无法加载模型，放弃当前案例")

        try:
            geo = sample["geometry"]
            load = sample["loading"]
            mat = sample["material"]

            # ---- 参数注入 ----
            model.parameter('c', f"{geo['c']} [mm]")
            model.parameter('sx', f"{geo['sx']} [mm]")
            model.parameter('sy', f"{geo['sy']} [mm]")
            model.parameter('r1', f"{geo['r1']} [mm]")
            model.parameter('a2', f"{geo['a2']} [mm]")
            model.parameter('b1', f"{geo['b1']} [mm]")
            model.parameter('b2', f"{geo['b2']} [mm]")

            model.parameter('A', f"{load['A']} [mm]")
            model.parameter('Ts', f"{load['Ts']} [s]")

            model.parameter('mu_0', f"{mat['mu']} [Pa*s]")

            print("   参数注入成功。正在重新剖分网格并求解...")

            # ---- 核心计算 ----
            model.mesh()
            model.solve()

            # ---- 导出 VTU 结果 ----
            export_node = model.java.result().export("data1")
            export_node.set("filename", str(vtu_filename))
            export_node.run()

            cost_time = time.time() - start_time
            print(f"✅ {case_id}.vtu 成功导出! 耗时: {cost_time:.1f} 秒")

        except Exception as e:
            print(f"❌ {case_id} 发生底层错误，完整报错信息如下：")
            traceback.print_exc()
            with open(error_log_path, "a", encoding='utf-8') as f_err:
                f_err.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {case_id} 失败:\n")
                f_err.write(traceback.format_exc() + "\n" + "-"*50 + "\n")
        finally:
            if model:
                model.clear()

    print(f"\n🎉 批处理结束！共尝试运行了 {run_count} 个案例。")
    client.disconnect()


if __name__ == "__main__":
    # 示例1：运行案例编号 51 到 199（即 Case_0051 到 Case_0199，共149个）
    # run_comsol_batch(start_case=51, end_case=200)

    # 示例2：从案例编号 100 开始，运行 50 个案例
    # run_comsol_batch(start_case=100, max_samples=50)

    # 示例3：只运行前 200 个案例（编号 1 到 200）
    run_comsol_batch(start_case=1, end_case=201)

    # 若想运行全部案例：run_comsol_batch(start_case=1)