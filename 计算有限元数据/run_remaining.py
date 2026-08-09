"""Automatically finish all missing COMSOL cases in isolated batches."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from failure_registry import (
    REGISTRY_PATH,
    load_failure_registry,
    synchronize_failure_registry,
)
from transfer2hdf5 import is_valid_hdf5


SCRIPT_DIR = Path(__file__).parent.resolve()
WORKER_PATH = SCRIPT_DIR / "main.py"
CONVERTER_PATH = SCRIPT_DIR / "transfer2hdf5.py"


def configure_workspace(
    workspace_root: Path | None = None,
    model_path: Path | None = None,
) -> None:
    """Point controller state and output at one isolated workspace."""
    global WORKSPACE_ROOT, PARAMETERS_PATH, MODEL_PATH, VTU_DIR, HDF5_DIR
    global LOG_DIR, STATE_PATH, REGISTRY_PATH
    WORKSPACE_ROOT = (
        SCRIPT_DIR if workspace_root is None else Path(workspace_root).resolve()
    )
    PARAMETERS_PATH = WORKSPACE_ROOT / "4_Combined_Master_Dataset.json"
    MODEL_PATH = (
        SCRIPT_DIR / "standard_model.mph"
        if model_path is None
        else Path(model_path).resolve()
    )
    VTU_DIR = WORKSPACE_ROOT / "comsol_output"
    HDF5_DIR = WORKSPACE_ROOT / "comsol_hdf5"
    LOG_DIR = WORKSPACE_ROOT / "batch_logs"
    STATE_PATH = WORKSPACE_ROOT / "batch_state.json"
    REGISTRY_PATH = WORKSPACE_ROOT / "failed_cases.json"


configure_workspace()


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"controller_{timestamp}_{os.getpid()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(path, encoding="utf-8"),
        ],
        force=True,
    )
    return path


def load_case_ids(path: Path | None = None) -> list[str]:
    path = PARAMETERS_PATH if path is None else Path(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    samples = payload.get("parameters_list")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"参数文件缺少非空 parameters_list: {path}")
    case_ids = [sample["case_id"] for sample in samples]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"参数文件包含重复 case_id: {', '.join(duplicates)}")
    return case_ids


def select_case_range(
    case_ids: list[str], first_case: int | None, last_case: int | None
) -> list[str]:
    first = 1 if first_case is None else first_case
    last = len(case_ids) if last_case is None else last_case
    if first < 1 or last > len(case_ids) or first > last:
        raise ValueError(f"工况范围必须满足 1 <= first <= last <= {len(case_ids)}")
    return case_ids[first - 1 : last]


def is_valid_vtu(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            tail = stream.read()
        return b"</VTKFile>" in tail
    except OSError:
        return False


def hdf5_complete(case_id: str) -> bool:
    return is_valid_hdf5(HDF5_DIR / f"{case_id}.h5")


def vtu_complete(case_id: str) -> bool:
    return is_valid_vtu(VTU_DIR / f"{case_id}.vtu")


def scan_completion(case_ids: list[str]) -> tuple[set[str], set[str]]:
    """Return valid HDF5 IDs and VTU-only IDs with one filesystem scan."""
    hdf5_done: set[str] = set()
    vtu_only: set[str] = set()
    for case_id in case_ids:
        if hdf5_complete(case_id):
            hdf5_done.add(case_id)
        elif vtu_complete(case_id):
            vtu_only.add(case_id)
    return hdf5_done, vtu_only


def sync_failed_cases(case_ids: list[str]) -> set[str]:
    hdf5_done, vtu_only = scan_completion(case_ids)
    return synchronize_failure_registry(
        case_ids,
        hdf5_done | vtu_only,
        log_dir=LOG_DIR,
        registry_path=REGISTRY_PATH,
    )


def compute_pending_cases(
    case_ids: list[str],
    hdf5_done: set[str],
    vtu_only: set[str],
    failed_cases: set[str],
    retry_failed: bool = False,
) -> list[str]:
    return [
        case_id
        for case_id in case_ids
        if case_id not in hdf5_done
        and case_id not in vtu_only
        and (retry_failed or case_id not in failed_cases)
    ]


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_worker_command(case_ids: list[str], cores: int) -> list[str]:
    return [
        sys.executable,
        str(WORKER_PATH),
        "--workspace-root",
        str(WORKSPACE_ROOT),
        "--model-path",
        str(MODEL_PATH),
        "--case-ids",
        ",".join(case_ids),
        "--cores",
        str(cores),
    ]


def build_converter_command(case_ids: list[str]) -> list[str]:
    return [
        sys.executable,
        str(CONVERTER_PATH),
        "--input-dir",
        str(VTU_DIR),
        "--output-dir",
        str(HDF5_DIR),
        "--case-ids",
        ",".join(case_ids),
    ]


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def run_worker(
    case_ids: list[str],
    cores: int,
    timeout_minutes: float,
    show_window: bool,
) -> int:
    command = build_worker_command(case_ids, cores)
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_CONSOLE
            if show_window
            else subprocess.CREATE_NO_WINDOW
        )

    logging.info("启动独立 worker: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=WORKSPACE_ROOT,
        creationflags=creationflags,
    )
    timeout = None if timeout_minutes <= 0 else timeout_minutes * 60
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.error("worker PID=%d 超过 %.1f 分钟，终止整个进程树", process.pid, timeout_minutes)
        kill_process_tree(process.pid)
        process.wait()
        return 124
    except KeyboardInterrupt:
        logging.warning("收到中断，正在关闭当前 worker 进程树")
        kill_process_tree(process.pid)
        process.wait()
        raise


def run_converter(case_ids: list[str]) -> int:
    if not case_ids:
        return 0
    logging.info("转换为 HDF5: %s", ", ".join(case_ids))
    completed = subprocess.run(
        build_converter_command(case_ids),
        cwd=WORKSPACE_ROOT,
        check=False,
    )
    return completed.returncode


def write_state(
    status: str,
    all_case_ids: list[str],
    current_batch: list[str] | None = None,
    pass_number: int = 0,
) -> None:
    completed_set, vtu_only_set = scan_completion(all_case_ids)
    failed_set = set(load_failure_registry(REGISTRY_PATH)["cases"])
    failed_set.difference_update(completed_set | vtu_only_set)
    completed = [case_id for case_id in all_case_ids if case_id in completed_set]
    vtu_only = [case_id for case_id in all_case_ids if case_id in vtu_only_set]
    failed = [case_id for case_id in all_case_ids if case_id in failed_set]
    pending_comsol = [
        case_id
        for case_id in all_case_ids
        if case_id not in completed_set
        and case_id not in vtu_only_set
        and case_id not in failed_set
    ]
    unresolved = [
        case_id
        for case_id in all_case_ids
        if case_id not in completed_set and case_id not in vtu_only_set
    ]
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "python": sys.executable,
        "pass_number": pass_number,
        "current_batch": current_batch or [],
        "total": len(all_case_ids),
        "hdf5_completed": len(completed),
        "vtu_waiting_for_conversion": vtu_only,
        "failed_terminal": failed,
        "pending_comsol": pending_comsol,
        "unresolved": unresolved,
    }
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, STATE_PATH)


def convert_pending_vtu(case_ids: list[str], batch_size: int) -> None:
    _, vtu_only = scan_completion(case_ids)
    pending = [case_id for case_id in case_ids if case_id in vtu_only]
    for batch in chunked(pending, batch_size):
        return_code = run_converter(batch)
        if return_code:
            logging.error("HDF5 转换子进程返回 %d", return_code)


def unresolved_cases(case_ids: list[str], convert: bool) -> list[str]:
    hdf5_done, vtu_only = scan_completion(case_ids)
    if convert:
        return [case_id for case_id in case_ids if case_id not in hdf5_done]
    return [
        case_id
        for case_id in case_ids
        if case_id not in hdf5_done and case_id not in vtu_only
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按独立进程批次自动续算全部缺失 COMSOL 工况"
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=10.0)
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=0.0,
        help="单批超时；0 表示不限时",
    )
    parser.add_argument("--first-case", type=int, help="首个工况序号，包含")
    parser.add_argument("--last-case", type=int, help="末个工况序号，包含")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-worker-window",
        action="store_true",
        help="后台运行 worker，不显示每批独立终端窗口",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="只生成 VTU，不自动转换 HDF5",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="显式重新计算 failed_cases.json 中的失败工况；默认永久跳过",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    if args.cores <= 0:
        raise ValueError("cores 必须大于 0")
    if args.max_retries < 0:
        raise ValueError("max-retries 不能小于 0")
    if args.pause_seconds < 0 or args.timeout_minutes < 0:
        raise ValueError("等待和超时时间不能小于 0")
    if Path(sys.prefix).name.lower() != "pinn":
        raise RuntimeError(
            f"当前不是 conda pinn 环境: {sys.executable}\n"
            "请先 conda activate pinn，或运行 运行剩余工况.bat。"
        )


def main() -> int:
    args = parse_args()
    try:
        configure_workspace(args.workspace_root, args.model_path)
        validate_args(args)
        all_case_ids = select_case_range(
            load_case_ids(), args.first_case, args.last_case
        )
    except Exception as error:
        print(f"参数错误: {error}", file=sys.stderr)
        return 2

    log_path = configure_logging()
    convert = not args.no_convert
    logging.info("控制器 PID=%d，Python=%s", os.getpid(), sys.executable)
    logging.info("日志=%s", log_path)
    logging.info(
        "范围=%s..%s，共 %d 个；batch=%d，最多尝试=%d 次",
        all_case_ids[0],
        all_case_ids[-1],
        len(all_case_ids),
        args.batch_size,
        args.max_retries + 1,
    )

    if convert and not args.dry_run:
        convert_pending_vtu(all_case_ids, args.batch_size)

    initial_pending = unresolved_cases(all_case_ids, convert)
    initial_hdf5, initial_vtu_only = scan_completion(all_case_ids)
    failed_cases = sync_failed_cases(all_case_ids)
    compute_pending = compute_pending_cases(
        all_case_ids,
        initial_hdf5,
        initial_vtu_only,
        failed_cases,
        retry_failed=args.retry_failed,
    )
    skipped_failed = (
        []
        if args.retry_failed
        else [case_id for case_id in all_case_ids if case_id in failed_cases]
    )
    logging.info(
        "当前 HDF5 完成=%d，已失败且默认跳过=%d，需 COMSOL 计算=%d，最终缺少 HDF5=%d",
        len(initial_hdf5),
        len(skipped_failed),
        len(compute_pending),
        len(initial_pending),
    )
    if skipped_failed:
        logging.info("跳过已记录失败工况: %s", ", ".join(skipped_failed))
    if args.retry_failed and failed_cases:
        logging.warning(
            "已启用 --retry-failed，本次会重新计算 %d 个历史失败工况",
            len(failed_cases),
        )
    if args.dry_run:
        for number, batch in enumerate(chunked(compute_pending, args.batch_size), 1):
            logging.info("dry-run 批次 %d: %s", number, ", ".join(batch))
        return 0

    try:
        for pass_number in range(1, args.max_retries + 2):
            if convert and pass_number > 1:
                convert_pending_vtu(all_case_ids, args.batch_size)
            remaining_before_pass = unresolved_cases(all_case_ids, convert)
            if not remaining_before_pass:
                break
            hdf5_done, vtu_only = scan_completion(all_case_ids)
            failed_cases = sync_failed_cases(all_case_ids)
            compute_pending = compute_pending_cases(
                all_case_ids,
                hdf5_done,
                vtu_only,
                failed_cases,
                retry_failed=args.retry_failed,
            )
            if not compute_pending:
                conversion_waiting = (
                    [case_id for case_id in all_case_ids if case_id in vtu_only]
                    if convert
                    else []
                )
                terminal_failed = [
                    case_id
                    for case_id in all_case_ids
                    if case_id in failed_cases
                    and case_id not in hdf5_done
                    and case_id not in vtu_only
                ]
                if terminal_failed and not conversion_waiting:
                    logging.info(
                        "只剩 %d 个已记录失败工况，不再自动重试",
                        len(terminal_failed),
                    )
                    break
                logging.warning(
                    "第 %d 次尝试只剩 HDF5 转换失败项，将进入下一次重试",
                    pass_number,
                )
                if args.pause_seconds:
                    time.sleep(args.pause_seconds)
                continue

            batches = list(chunked(compute_pending, args.batch_size))
            logging.info(
                "第 %d 次尝试：%d 个 COMSOL 工况，分 %d 批",
                pass_number,
                len(compute_pending),
                len(batches),
            )
            for batch_number, batch in enumerate(batches, 1):
                write_state("running", all_case_ids, batch, pass_number)
                logging.info(
                    "批次 %d/%d 启动: %s",
                    batch_number,
                    len(batches),
                    ", ".join(batch),
                )
                return_code = run_worker(
                    batch,
                    args.cores,
                    args.timeout_minutes,
                    show_window=not args.no_worker_window,
                )
                logging.info(
                    "worker 已完全退出，独立终端已关闭，返回码=%d", return_code
                )
                if return_code:
                    logging.warning(
                        "本批存在失败项；明确计算失败的工况将写入清单并停止自动重试"
                    )
                if convert:
                    converter_code = run_converter(batch)
                    if converter_code:
                        logging.warning("本批 HDF5 转换存在失败项")
                sync_failed_cases(all_case_ids)
                write_state("between_batches", all_case_ids, None, pass_number)
                if args.pause_seconds and batch_number < len(batches):
                    logging.info("等待 %.1f 秒后启动下一全新进程", args.pause_seconds)
                    time.sleep(args.pause_seconds)
    except KeyboardInterrupt:
        logging.warning("用户中断；已有原子输出保留，下次运行会自动续算")
        write_state("interrupted", all_case_ids)
        return 130
    except Exception:
        logging.exception("控制器发生未处理异常；已有原子输出仍可用于下次续算")
        try:
            write_state("controller_error", all_case_ids)
        except Exception:
            logging.exception("写入异常状态文件失败")
        return 2

    if convert:
        convert_pending_vtu(all_case_ids, args.batch_size)
    failed_cases = sync_failed_cases(all_case_ids)
    remaining = unresolved_cases(all_case_ids, convert)
    if remaining:
        terminal_failed = [case_id for case_id in remaining if case_id in failed_cases]
        retryable = [case_id for case_id in remaining if case_id not in failed_cases]
        logging.error(
            "仍缺少 %d 个 HDF5：已记录失败=%d，其他未完成=%d",
            len(remaining),
            len(terminal_failed),
            len(retryable),
        )
        if terminal_failed:
            logging.error("已记录失败工况: %s", ", ".join(terminal_failed))
        if retryable:
            logging.error("其他未完成工况: %s", ", ".join(retryable))
        status = "complete_with_failures" if not retryable else "incomplete"
        write_state(status, all_case_ids)
        return 1

    logging.info("所选范围全部完成")
    write_state("complete", all_case_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
