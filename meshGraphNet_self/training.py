import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from .dataset import CASE_FEATURE_NAMES, FpcDataset
    from .early_stopping import reference_from_checkpoint, update_early_stopping
    from .experiment_utils import (
        atomic_write_json,
        capture_rng_state,
        read_metrics_rows,
        restore_rng_state,
        select_manifest_cases,
        upsert_metrics_row,
    )
    from .graph import build_graph_transform, prepare_graph
except ImportError:
    from dataset import CASE_FEATURE_NAMES, FpcDataset
    from early_stopping import reference_from_checkpoint, update_early_stopping
    from experiment_utils import (
        atomic_write_json,
        capture_rng_state,
        read_metrics_rows,
        restore_rng_state,
        select_manifest_cases,
        upsert_metrics_row,
    )
    from graph import build_graph_transform, prepare_graph

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


FIELD_NAMES = ("p", "T")


def add_common_training_args(
    parser: argparse.ArgumentParser, project_root: Path, self_root: Path
) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "计算有限元数据" / "comsol_hdf5",
    )
    parser.add_argument("--parameters-json", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=self_root / "checkpoints")
    parser.add_argument("--log-dir", type=Path, default=self_root / "runs")
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch-log-every", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--early-stopping-min-relative-improvement", type=float, default=0.002
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-validate-mesh-domain", action="store_true")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    device,
    generator: Optional[torch.Generator] = None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    transform,
    device,
    gradient_clip,
    epoch,
    epochs,
    batch_loss_callback: Optional[Callable[[int, float], None]] = None,
):
    model.train()
    total_squared_error = 0.0
    total_values = 0
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=False)

    for batch_index, graph in enumerate(progress, start=1):
        graph = prepare_graph(graph, transform).to(device)
        predicted, target = model(graph)
        loss = F.mse_loss(predicted, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_squared_error += F.mse_loss(
            predicted.detach(), target.detach(), reduction="sum"
        ).item()
        total_values += target.numel()
        progress.set_postfix(loss=f"{loss.item():.3e}")
        if batch_loss_callback is not None:
            batch_loss_callback(batch_index, float(loss.item()))

    return total_squared_error / max(total_values, 1)


@torch.no_grad()
def evaluate(model, loader, transform, device, description="valid") -> Dict[str, float]:
    model.eval()
    normalized_squared_error = 0.0
    normalized_values = 0
    field_squared_error = torch.zeros(len(FIELD_NAMES), device=device)
    node_count = 0

    for graph in tqdm(loader, desc=description, leave=False):
        graph = prepare_graph(graph, transform).to(device)
        predicted_delta, target_delta = model.normalized_prediction_and_target(
            graph, accumulate=False
        )
        normalized_squared_error += F.mse_loss(
            predicted_delta, target_delta, reduction="sum"
        ).item()
        normalized_values += target_delta.numel()

        current_fields = graph.x[:, 1 : 1 + len(FIELD_NAMES)]
        predicted_next = current_fields + model.output_normalizer.inverse(
            predicted_delta
        )
        field_squared_error += (predicted_next - graph.y).square().sum(dim=0)
        node_count += graph.y.shape[0]

    metrics = {
        "normalized_mse": normalized_squared_error / max(normalized_values, 1)
    }
    rmse = torch.sqrt(field_squared_error / max(node_count, 1)).cpu().tolist()
    for name, value in zip(FIELD_NAMES, rmse):
        metrics[f"rmse_{name}"] = value
    return metrics


@torch.no_grad()
def fit_training_normalizers(model, loader, transform, device) -> dict:
    """Fit all feature/target statistics once from the training split."""
    model.reset_normalizers()
    for graph in tqdm(loader, desc="fit normalization", leave=False):
        graph = prepare_graph(graph, transform).to(device)
        model.accumulate_normalizers(graph)

    normalizers = model.normalizers()
    for name, normalizer in normalizers.items():
        if normalizer.acc_count.item() <= 0:
            raise RuntimeError(f"Normalizer {name!r} received no training values.")
        if not torch.isfinite(normalizer.mean).all():
            raise RuntimeError(f"Normalizer {name!r} has a non-finite mean.")
        if not torch.isfinite(normalizer.raw_std).all():
            raise RuntimeError(f"Normalizer {name!r} has a non-finite std.")

    # Physical fields and prediction targets must vary. Constant geometric
    # channels such as mesh_velocity_x are allowed and normalize to zero.
    for name in ("field", "output"):
        raw_std = normalizers[name].raw_std
        if torch.any(raw_std <= normalizers[name].std_epsilon):
            raise RuntimeError(
                f"Normalizer {name!r} contains an unexpectedly constant field: "
                f"std={raw_std.flatten().cpu().tolist()}"
            )

    model.freeze_normalizers()
    statistics = {
        name: {
            "count": int(normalizer.acc_count.item()),
            "mean": normalizer.mean.detach().cpu().flatten().tolist(),
            "std": normalizer.raw_std.detach().cpu().flatten().tolist(),
        }
        for name, normalizer in normalizers.items()
    }
    print(
        "normalization fitted and frozen: "
        f"field_mean={statistics['field']['mean']} "
        f"field_std={statistics['field']['std']} "
        f"output_mean={statistics['output']['mean']} "
        f"output_std={statistics['output']['std']}"
    )
    return statistics


def freeze_restored_normalizers(model) -> dict:
    """Freeze statistics restored from a legacy or current checkpoint."""
    normalizers = model.normalizers()
    for name, normalizer in normalizers.items():
        if normalizer.acc_count.item() <= 0:
            raise RuntimeError(
                f"Checkpoint normalizer {name!r} has no fitted statistics."
            )
        if not torch.isfinite(normalizer.mean).all() or not torch.isfinite(
            normalizer.raw_std
        ).all():
            raise RuntimeError(
                f"Checkpoint normalizer {name!r} contains non-finite statistics."
            )
        normalizer.freeze()
    for name in ("field", "output"):
        raw_std = normalizers[name].raw_std
        if torch.any(raw_std <= normalizers[name].std_epsilon):
            raise RuntimeError(
                f"Checkpoint normalizer {name!r} is numerically collapsed: "
                f"std={raw_std.flatten().cpu().tolist()}. Restart this run with "
                "precomputed normalization statistics."
            )
    return {
        name: {
            "count": int(normalizer.acc_count.item()),
            "mean": normalizer.mean.detach().cpu().flatten().tolist(),
            "std": normalizer.raw_std.detach().cpu().flatten().tolist(),
        }
        for name, normalizer in normalizers.items()
    }


def save_normalization_statistics(path: Path, model_name: str, statistics: dict) -> None:
    payload = {
        "version": 2,
        "algorithm": "float64_batch_welford",
        "source": "training_split_only",
        "frozen_during_optimization": True,
        "model_name": model_name,
        "normalizers": statistics,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def checkpoint_state(
    epoch,
    global_step,
    model,
    optimizer,
    scheduler,
    best_valid_loss,
    model_config,
    epochs_without_improvement=0,
    early_stopping_reference_loss=float("inf"),
    data_loader_generator=None,
    metrics_row=None,
):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "early_stopping_reference_loss": early_stopping_reference_loss,
        "model_config": model_config,
        "field_names": FIELD_NAMES,
        "case_feature_names": CASE_FEATURE_NAMES,
        "rng_state": capture_rng_state(data_loader_generator),
        "metrics_row": metrics_row,
    }


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def restore_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    device,
    data_loader_generator=None,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state"), data_loader_generator)
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_valid_loss", float("inf"))),
        int(checkpoint.get("epochs_without_improvement", 0)),
        reference_from_checkpoint(
            checkpoint,
            float(checkpoint.get("best_valid_loss", float("inf"))),
        ),
        checkpoint.get("metrics_row"),
    )


def _dataset_case_ids(args):
    if args.split_manifest is None:
        if args.train_size is not None:
            raise ValueError("--train-size requires --split-manifest.")
        return None, None, None, None
    return select_manifest_cases(args.split_manifest, args.train_size)


def _build_datasets(args):
    train_ids, valid_ids, test_ids, manifest = _dataset_case_ids(args)
    dataset_kwargs = {
        "data_root": str(args.data_root),
        "parameters_json": (
            str(args.parameters_json) if args.parameters_json is not None else None
        ),
        "validate_mesh_domain": not args.no_validate_mesh_domain,
        "field_names": FIELD_NAMES,
    }
    train_dataset = FpcDataset(
        split="train", case_ids=train_ids, **dataset_kwargs
    )
    valid_dataset = FpcDataset(
        split="valid", case_ids=valid_ids, **dataset_kwargs
    )
    test_dataset = None
    if args.evaluate_test:
        test_dataset = FpcDataset(
            split="test", case_ids=test_ids, **dataset_kwargs
        )
    return train_dataset, valid_dataset, test_dataset, manifest


def run_training(
    args,
    *,
    model_config: dict,
    model_factory: Callable[[], torch.nn.Module],
    model_name: str,
    extra_config: Optional[dict] = None,
    extra_checkpoint: Optional[dict] = None,
) -> dict:
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        args.metrics_file
        if args.metrics_file is not None
        else args.checkpoint_dir / "metrics.csv"
    )
    summary_path = (
        args.summary_file
        if args.summary_file is not None
        else args.checkpoint_dir / "summary.json"
    )

    train_dataset, valid_dataset, test_dataset, manifest = _build_datasets(args)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = create_dataloader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        device,
        generator=train_generator,
    )
    normalization_loader = create_dataloader(
        train_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
    )
    valid_loader = create_dataloader(
        valid_dataset, args.batch_size, False, args.num_workers, device
    )
    test_loader = (
        create_dataloader(
            test_dataset, args.batch_size, False, args.num_workers, device
        )
        if test_dataset is not None
        else None
    )

    model = model_factory().to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    transform = build_graph_transform()

    start_epoch = 1
    global_step = 0
    best_valid_loss = float("inf")
    early_stopping_reference_loss = float("inf")
    epochs_without_improvement = 0
    restored_metrics = None
    if args.resume is not None:
        (
            start_epoch,
            global_step,
            best_valid_loss,
            epochs_without_improvement,
            early_stopping_reference_loss,
            restored_metrics,
        ) = restore_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
            data_loader_generator=train_generator,
        )
        if restored_metrics is not None:
            upsert_metrics_row(metrics_path, restored_metrics)

        normalization_statistics = freeze_restored_normalizers(model)
        print("restored normalization statistics and froze them for resumed training")
    else:
        normalization_statistics = fit_training_normalizers(
            model, normalization_loader, transform, device
        )
    save_normalization_statistics(
        args.checkpoint_dir / "normalization_stats.pt",
        model_name,
        normalization_statistics,
    )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "model_name": model_name,
            "model_config": model_config,
            "train_case_count": len(train_dataset.files),
            "valid_case_count": len(valid_dataset.files),
            "test_case_count": len(test_dataset.files) if test_dataset else 0,
            "split_manifest_resolved": manifest.get("_path") if manifest else None,
        }
    )
    if extra_config:
        config.update(extra_config)
    atomic_write_json(args.checkpoint_dir / "training_config.json", config)

    writer: Optional[object] = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(args.log_dir), flush_secs=10)
    else:
        print("TensorBoard is unavailable; continuing without event logs.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} model={model_name} parameters={parameter_count:,} "
        f"train_cases={len(train_dataset.files)} "
        f"valid_cases={len(valid_dataset.files)} "
        f"train_graphs={len(train_dataset)} valid_graphs={len(valid_dataset)}"
    )

    stopped_early = bool(
        args.early_stopping_patience > 0
        and epochs_without_improvement >= args.early_stopping_patience
    )
    if stopped_early:
        print(
            "checkpoint already satisfies early-stopping condition; "
            "skipping remaining epochs"
        )
    try:
        epoch_range = (
            range(0) if stopped_early else range(start_epoch, args.epochs + 1)
        )
        for epoch in epoch_range:
            epoch_start = time.perf_counter()
            epoch_start_step = global_step

            def record_batch_loss(batch_index: int, batch_loss: float) -> None:
                if writer is None or args.batch_log_every <= 0:
                    return
                step = epoch_start_step + batch_index
                if step % args.batch_log_every == 0 or batch_index == len(train_loader):
                    writer.add_scalar(
                        "loss/train_batch_normalized_mse", batch_loss, step
                    )

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                transform,
                device,
                args.gradient_clip,
                epoch,
                args.epochs,
                batch_loss_callback=record_batch_loss,
            )
            global_step += len(train_loader)
            valid_metrics = evaluate(model, valid_loader, transform, device)
            valid_loss = valid_metrics["normalized_mse"]
            scheduler.step(valid_loss)

            early_stopping_update = update_early_stopping(
                valid_loss,
                best_valid_loss,
                early_stopping_reference_loss,
                epochs_without_improvement,
                args.early_stopping_min_relative_improvement,
            )
            best_valid_loss = early_stopping_update.best_valid_loss
            early_stopping_reference_loss = early_stopping_update.reference_loss
            epochs_without_improvement = (
                early_stopping_update.epochs_without_improvement
            )
            improved = early_stopping_update.exact_improvement

            epoch_seconds = time.perf_counter() - epoch_start
            metrics_row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_normalized_mse": train_loss,
                "valid_normalized_mse": valid_loss,
                "valid_rmse_p": valid_metrics["rmse_p"],
                "valid_rmse_T": valid_metrics["rmse_T"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "early_stop_wait": epochs_without_improvement,
                "relative_improvement": early_stopping_update.relative_improvement,
                "meaningful_improvement": (
                    early_stopping_update.meaningful_improvement
                ),
                "epoch_seconds": epoch_seconds,
            }
            print(
                f"epoch={epoch:04d} train_mse={train_loss:.4e} "
                f"valid_mse={valid_loss:.4e} "
                f"p_rmse={valid_metrics['rmse_p']:.4e} "
                f"T_rmse={valid_metrics['rmse_T']:.4e} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"early_stop_wait={epochs_without_improvement}/"
                f"{args.early_stopping_patience} "
                f"relative_improvement="
                f"{early_stopping_update.relative_improvement:.3%} "
                f"meaningful_improvement="
                f"{early_stopping_update.meaningful_improvement} "
                f"seconds={epoch_seconds:.1f}"
            )

            if writer is not None:
                writer.add_scalar("loss/train_normalized_mse", train_loss, epoch)
                writer.add_scalar("loss/valid_normalized_mse", valid_loss, epoch)
                writer.add_scalar("rmse/p", valid_metrics["rmse_p"], epoch)
                writer.add_scalar("rmse/T", valid_metrics["rmse_T"], epoch)
                writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], epoch
                )
                writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)
                writer.add_scalar(
                    "early_stopping/wait", epochs_without_improvement, epoch
                )
                writer.add_scalar(
                    "early_stopping/relative_improvement",
                    early_stopping_update.relative_improvement,
                    epoch,
                )
                writer.flush()

            state = checkpoint_state(
                epoch,
                global_step,
                model,
                optimizer,
                scheduler,
                best_valid_loss,
                model_config,
                epochs_without_improvement=epochs_without_improvement,
                early_stopping_reference_loss=early_stopping_reference_loss,
                data_loader_generator=train_generator,
                metrics_row=metrics_row,
            )
            state["model_name"] = model_name
            if extra_checkpoint:
                state.update(extra_checkpoint)
            save_checkpoint(state, args.checkpoint_dir / "last.pt")
            if improved:
                save_checkpoint(state, args.checkpoint_dir / "best.pt")
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    state, args.checkpoint_dir / f"epoch_{epoch:04d}.pt"
                )
            upsert_metrics_row(metrics_path, metrics_row)

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                print(
                    f"early stopping at epoch={epoch}; "
                    f"patience={args.early_stopping_patience}; "
                    f"min_relative_improvement="
                    f"{args.early_stopping_min_relative_improvement:.3%}"
                )
                break
    finally:
        if writer is not None:
            writer.close()

    best_path = args.checkpoint_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training finished without creating best.pt.")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            model, test_loader, transform, device, description="test"
        )
        print(
            f"test_mse={test_metrics['normalized_mse']:.4e} "
            f"test_p_rmse={test_metrics['rmse_p']:.4e} "
            f"test_T_rmse={test_metrics['rmse_T']:.4e}"
        )

    metric_rows = read_metrics_rows(metrics_path)
    total_epoch_seconds = sum(
        float(row.get("epoch_seconds", 0.0) or 0.0) for row in metric_rows
    )
    final_epoch = max((int(row["epoch"]) for row in metric_rows), default=0)
    summary = {
        "completed": True,
        "model_name": model_name,
        "parameter_count": parameter_count,
        "seed": args.seed,
        "train_size": len(train_dataset.files),
        "valid_size": len(valid_dataset.files),
        "test_size": len(test_dataset.files) if test_dataset else 0,
        "train_graphs": len(train_dataset),
        "valid_graphs": len(valid_dataset),
        "test_graphs": len(test_dataset) if test_dataset else 0,
        "final_epoch": final_epoch,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_valid_normalized_mse": float(
            best_checkpoint["best_valid_loss"]
        ),
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_relative_improvement": (
            args.early_stopping_min_relative_improvement
        ),
        "early_stopping_reference_loss": early_stopping_reference_loss,
        "early_stopping_final_wait": epochs_without_improvement,
        "total_epoch_seconds": total_epoch_seconds,
        "metrics_file": str(Path(metrics_path).resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "split_manifest": manifest.get("_path") if manifest else None,
    }
    if test_metrics is not None:
        summary.update(
            {
                "test_normalized_mse": test_metrics["normalized_mse"],
                "test_rmse_p": test_metrics["rmse_p"],
                "test_rmse_T": test_metrics["rmse_T"],
            }
        )
    atomic_write_json(summary_path, summary)
    return summary
