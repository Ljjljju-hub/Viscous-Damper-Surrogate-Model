import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from .dataset import CASE_FEATURE_NAMES, FpcDataset
    from .graph import build_graph_transform, prepare_graph
    from .model.simulator import SurrogateSimulator
except ImportError:
    from dataset import CASE_FEATURE_NAMES, FpcDataset
    from graph import build_graph_transform, prepare_graph
    from model.simulator import SurrogateSimulator


FIELD_NAMES = ("p", "T")


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    self_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the viscous-damper MeshGraphNet.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "计算有限元数据" / "comsol_hdf5",
    )
    parser.add_argument("--parameters-json", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=self_root / "checkpoints")
    parser.add_argument("--log-dir", type=Path, default=self_root / "runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--message-passing-steps", type=int, default=15)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-validate-mesh-domain", action="store_true")
    return parser.parse_args()


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


def create_dataloader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
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
):
    model.train()
    total_squared_error = 0.0
    total_values = 0
    progress = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=False)

    for graph in progress:
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

    return total_squared_error / max(total_values, 1)


@torch.no_grad()
def evaluate(model, loader, transform, device) -> Dict[str, float]:
    model.eval()
    normalized_squared_error = 0.0
    normalized_values = 0
    field_squared_error = torch.zeros(len(FIELD_NAMES), device=device)
    node_count = 0

    for graph in tqdm(loader, desc="valid", leave=False):
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


def checkpoint_state(
    epoch,
    global_step,
    model,
    optimizer,
    scheduler,
    best_valid_loss,
    model_config,
):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "model_config": model_config,
        "field_names": FIELD_NAMES,
        "case_feature_names": CASE_FEATURE_NAMES,
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
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_valid_loss", float("inf"))),
    )


def main(args=None):
    args = parse_args() if args is None else args
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {
        "data_root": str(args.data_root),
        "parameters_json": (
            str(args.parameters_json) if args.parameters_json is not None else None
        ),
        "validate_mesh_domain": not args.no_validate_mesh_domain,
        "field_names": FIELD_NAMES,
    }
    train_dataset = FpcDataset(split="train", **dataset_kwargs)
    valid_dataset = FpcDataset(split="valid", **dataset_kwargs)
    train_loader = create_dataloader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    valid_loader = create_dataloader(
        valid_dataset, args.batch_size, False, args.num_workers, device
    )

    model_config = {
        "field_count": len(FIELD_NAMES),
        "case_feature_count": len(CASE_FEATURE_NAMES),
        "region_count": 3,
        "edge_input_size": 3,
        "hidden_size": args.hidden_size,
        "message_passing_steps": args.message_passing_steps,
    }
    model = SurrogateSimulator(**model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    transform = build_graph_transform()

    start_epoch = 1
    global_step = 0
    best_valid_loss = float("inf")
    if args.resume is not None:
        start_epoch, global_step, best_valid_loss = restore_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    config_path = args.checkpoint_dir / "training_config.json"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["model_config"] = model_config
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    writer: Optional[object] = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(args.log_dir))
    else:
        print("TensorBoard is unavailable; continuing without event logs.")

    print(
        f"device={device} train_graphs={len(train_dataset)} "
        f"valid_graphs={len(valid_dataset)}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            transform,
            device,
            args.gradient_clip,
            epoch,
            args.epochs,
        )
        global_step += len(train_loader)
        valid_metrics = evaluate(model, valid_loader, transform, device)
        valid_loss = valid_metrics["normalized_mse"]
        scheduler.step(valid_loss)

        print(
            f"epoch={epoch:04d} train_mse={train_loss:.4e} "
            f"valid_mse={valid_loss:.4e} "
            f"p_rmse={valid_metrics['rmse_p']:.4e} "
            f"T_rmse={valid_metrics['rmse_T']:.4e} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        if writer is not None:
            writer.add_scalar("loss/train_normalized_mse", train_loss, epoch)
            writer.add_scalar("loss/valid_normalized_mse", valid_loss, epoch)
            writer.add_scalar("rmse/p", valid_metrics["rmse_p"], epoch)
            writer.add_scalar("rmse/T", valid_metrics["rmse_T"], epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        state = checkpoint_state(
            epoch,
            global_step,
            model,
            optimizer,
            scheduler,
            min(best_valid_loss, valid_loss),
            model_config,
        )
        save_checkpoint(state, args.checkpoint_dir / "last.pt")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            state["best_valid_loss"] = best_valid_loss
            save_checkpoint(state, args.checkpoint_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(state, args.checkpoint_dir / f"epoch_{epoch:04d}.pt")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
