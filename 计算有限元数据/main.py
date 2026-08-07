"""Run one isolated batch of COMSOL cases.

This module is the worker process.  ``run_remaining.py`` starts a fresh Python
process for every batch so that the COMSOL JVM and server never survive across
batches.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import h5py
import mph
import mph.session as mph_session


BASE_DIR = Path(__file__).parent.resolve()
PARAMETERS_PATH = BASE_DIR / "4_Combined_Master_Dataset.json"
MODEL_PATH = BASE_DIR / "standard_model.mph"
VTU_DIR = BASE_DIR / "comsol_output"
HDF5_DIR = BASE_DIR / "comsol_hdf5"
LOG_DIR = BASE_DIR / "batch_logs"


def configure_logging(case_ids: list[str]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    first = case_ids[0] if case_ids else "empty"
    last = case_ids[-1] if case_ids else "empty"
    log_path = LOG_DIR / f"worker_{first}_{last}_{timestamp}_{os.getpid()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def load_samples(path: Path = PARAMETERS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    samples = payload.get("parameters_list")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"参数文件缺少非空 parameters_list: {path}")
    return samples


def select_samples(
    samples: list[dict],
    case_ids: Iterable[str] | None = None,
    start_case: int = 1,
    end_case: int | None = None,
    max_samples: int | None = None,
) -> list[dict]:
    """Select explicit IDs or a backwards-compatible 1-based index range."""
    if case_ids:
        sample_by_id = {sample["case_id"]: sample for sample in samples}
        requested = list(dict.fromkeys(case_ids))
        unknown = [case_id for case_id in requested if case_id not in sample_by_id]
        if unknown:
            raise ValueError(f"参数 JSON 中不存在工况: {', '.join(unknown)}")
        return [sample_by_id[case_id] for case_id in requested]

    if start_case < 1 or start_case > len(samples):
        raise ValueError(f"start_case 必须位于 1..{len(samples)}")
    start_idx = start_case - 1
    if end_case is not None:
        end_idx = min(end_case - 1, len(samples))
    elif max_samples is not None:
        end_idx = min(start_idx + max_samples, len(samples))
    else:
        end_idx = len(samples)
    if end_idx <= start_idx:
        raise ValueError("所选工况范围为空")
    return samples[start_idx:end_idx]


def valid_hdf5(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    try:
        with h5py.File(path, "r") as handle:
            coordinates = handle["mesh/coordinates"]
            connectivity = handle["mesh/connectivity"]
            time_steps = handle["time_steps"]
            fields = handle["fields"]
            return (
                coordinates.ndim == 2
                and coordinates.shape[0] > 0
                and connectivity.size > 0
                and time_steps.ndim == 1
                and time_steps.size > 0
                and bool(fields.keys())
                and all(
                    dataset.shape == (time_steps.size, coordinates.shape[0])
                    for dataset in fields.values()
                )
            )
    except (OSError, KeyError, ValueError):
        return False


def valid_vtu(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            return b"</VTKFile>" in stream.read()
    except OSError:
        return False


def output_exists(case_id: str) -> bool:
    return valid_hdf5(HDF5_DIR / f"{case_id}.h5") or valid_vtu(
        VTU_DIR / f"{case_id}.vtu"
    )


def inject_parameters(model, sample: dict) -> None:
    geo = sample["geometry"]
    load = sample["loading"]
    material = sample["material"]

    for name in ("c", "sx", "sy", "r1", "a2", "b1", "b2"):
        model.parameter(name, f"{geo[name]} [mm]")
    model.parameter("A", f"{load['A']} [mm]")
    model.parameter("Ts", f"{load['Ts']} [s]")
    model.parameter("mu_0", f"{material['mu']} [Pa*s]")


def remove_model(client, model) -> None:
    if model is None:
        return
    try:
        client.remove(model)
    except Exception:
        logging.exception("卸载 COMSOL 模型失败，继续退出当前 worker")
        try:
            model.clear()
        except Exception:
            logging.exception("清理 COMSOL 模型失败")


def shutdown_comsol(client) -> None:
    """Disconnect the client and explicitly stop the MPh-owned server."""
    if client is not None:
        try:
            client.disconnect()
        except RuntimeError:
            # A stand-alone client has no server connection to disconnect.
            logging.info("COMSOL client 为 stand-alone 模式，将随 worker 进程退出")
        except Exception:
            logging.exception("断开 COMSOL client 时发生错误")

    server = mph_session.server
    if server is not None:
        try:
            if server.running():
                server.stop(timeout=30)
        except Exception:
            logging.exception("停止 MPh server 时发生错误，worker 退出后由操作系统回收")
    gc.collect()


def solve_case(client, sample: dict, force: bool = False) -> bool:
    case_id = sample["case_id"]
    target = VTU_DIR / f"{case_id}.vtu"
    partial = VTU_DIR / f".{case_id}.{os.getpid()}.partial.vtu"

    if not force and output_exists(case_id):
        logging.info("%s 已有有效输出，跳过", case_id)
        return True

    partial.unlink(missing_ok=True)
    model = None
    started = time.monotonic()
    try:
        logging.info("开始计算 %s", case_id)
        model = client.load(str(MODEL_PATH))
        inject_parameters(model, sample)
        model.mesh()
        model.solve()

        export_node = model.java.result().export("data1")
        export_node.set("filename", str(partial))
        export_node.run()
        if not partial.is_file() or partial.stat().st_size <= 1024:
            raise RuntimeError(f"COMSOL 未生成完整 VTU 临时文件: {partial}")
        os.replace(partial, target)
        logging.info("%s 完成，耗时 %.1f 秒", case_id, time.monotonic() - started)
        return True
    except Exception:
        logging.error("%s 计算失败\n%s", case_id, traceback.format_exc())
        partial.unlink(missing_ok=True)
        return False
    finally:
        remove_model(client, model)
        gc.collect()


def run_comsol_batch(
    case_ids: Iterable[str] | None = None,
    start_case: int = 1,
    end_case: int | None = None,
    max_samples: int | None = None,
    cores: int = 16,
    force: bool = False,
) -> list[str]:
    samples = select_samples(
        load_samples(), case_ids, start_case, end_case, max_samples
    )
    selected_ids = [sample["case_id"] for sample in samples]
    log_path = configure_logging(selected_ids)
    logging.info("worker PID=%s，日志=%s", os.getpid(), log_path)
    logging.info("本批工况数=%d: %s", len(samples), ", ".join(selected_ids))

    VTU_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到 COMSOL 母版模型: {MODEL_PATH}")

    client = None
    failed: list[str] = []
    try:
        # Client-server mode gives us an explicit server process that can be
        # stopped at the batch boundary.
        mph.option("session", "client-server")
        logging.info("启动 COMSOL client-server，会话使用 %d 核", cores)
        client = mph.start(cores=cores)
        for sample in samples:
            if not solve_case(client, sample, force=force):
                failed.append(sample["case_id"])
    finally:
        shutdown_comsol(client)
        logging.info("COMSOL 会话已关闭，worker 即将退出")

    if failed:
        logging.error("本批失败 %d 个: %s", len(failed), ", ".join(failed))
    else:
        logging.info("本批全部完成")
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行一个隔离的 COMSOL 工况批次")
    parser.add_argument(
        "--case-ids",
        help="逗号分隔的 case_id；由 run_remaining.py 自动传入",
    )
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--end-case", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="覆盖已有 VTU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_ids = None
    if args.case_ids:
        case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    try:
        failed = run_comsol_batch(
            case_ids=case_ids,
            start_case=args.start_case,
            end_case=args.end_case,
            max_samples=args.max_samples,
            cores=args.cores,
            force=args.force,
        )
    except Exception:
        logging.basicConfig(level=logging.INFO, force=False)
        logging.exception("worker 启动或运行失败")
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
