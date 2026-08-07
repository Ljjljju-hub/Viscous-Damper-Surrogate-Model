import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.dataset import CASE_FEATURE_NAMES, FpcDataset
from meshGraphNet_self.graph import build_graph_transform
from meshGraphNet_self.train import (
    FIELD_NAMES,
    checkpoint_state,
    choose_device,
    create_dataloader,
    evaluate,
    restore_checkpoint,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from transolver_self.model.simulator import TransolverSimulator

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


OFFICIAL_TRANSOLVER_REVISION = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"


def parse_args():
    self_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Transolver on the viscous-damper moving mesh."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "计算有限元数据" / "comsol_hdf5",
    )
    parser.add_argument("--parameters-json", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=self_root / "checkpoints")
    parser.add_argument("--log-dir", type=Path, default=self_root / "runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-ratio", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-validate-mesh-domain", action="store_true")
    return parser.parse_args()


def build_model_config(args) -> dict:
    return {
        "field_count": len(FIELD_NAMES),
        "case_feature_count": len(CASE_FEATURE_NAMES),
        "region_count": 3,
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "heads": args.heads,
        "slice_num": args.slice_num,
        "dropout": args.dropout,
        "mlp_ratio": args.mlp_ratio,
    }


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

    model_config = build_model_config(args)
    model = TransolverSimulator(**model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    # Reusing the MeshGraphNet train/evaluate functions keeps target and metrics exact.
    transform = build_graph_transform()

    start_epoch = 1
    global_step = 0
    best_valid_loss = float("inf")
    if args.resume is not None:
        start_epoch, global_step, best_valid_loss = restore_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "model": "THUML Transolver irregular mesh",
            "official_revision": OFFICIAL_TRANSOLVER_REVISION,
            "model_config": model_config,
        }
    )
    (args.checkpoint_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    writer: Optional[object] = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(args.log_dir))
    else:
        print("TensorBoard is unavailable; continuing without event logs.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count:,} "
        f"train_graphs={len(train_dataset)} valid_graphs={len(valid_dataset)}"
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
        state["model_name"] = "transolver"
        state["official_revision"] = OFFICIAL_TRANSOLVER_REVISION
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
