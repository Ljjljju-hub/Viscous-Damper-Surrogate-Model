import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.dataset import FpcDataset
from meshGraphNet_self.training import FIELD_NAMES, choose_device
from transolver_self.model.simulator import TransolverSimulator


def load_model(checkpoint_path: Path, device: torch.device) -> TransolverSimulator:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("model_name") != "transolver":
        raise ValueError(f"{checkpoint_path} is not a Transolver checkpoint.")
    model = TransolverSimulator(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def attach_fields(mesh: Data, fields: torch.Tensor) -> Data:
    node_type = torch.zeros(
        (fields.shape[0], 1), dtype=fields.dtype, device=fields.device
    )
    mesh.x = torch.cat([node_type, fields], dim=-1)
    mesh.predicted_fields = fields
    mesh.p = fields[:, 0:1]
    mesh.T = fields[:, 1:2]
    return mesh


@torch.no_grad()
def rollout_case(
    dataset: FpcDataset,
    model: TransolverSimulator,
    case_id: str,
    device: torch.device,
    start_index: int = 0,
    steps: int | None = None,
) -> Tuple[List[Data], Dict[str, float]]:
    file_path = dataset._resolve_file(case_id)
    with h5py.File(str(file_path), "r") as h5_file:
        times = torch.as_tensor(h5_file["time_steps"][:], dtype=torch.float32)
        truth = torch.stack(
            [
                torch.as_tensor(h5_file[f"fields/{name}"][:], dtype=torch.float32)
                for name in FIELD_NAMES
            ],
            dim=-1,
        )

    available_steps = len(times) - 1 - start_index
    if start_index < 0 or available_steps < 0:
        raise ValueError("start_index is outside the stored time range.")
    rollout_steps = available_steps if steps is None else min(steps, available_steps)
    if rollout_steps < 1:
        raise ValueError("The rollout must contain at least one prediction step.")

    current_fields = truth[start_index].to(device)
    meshes: List[Data] = []
    initial_mesh = dataset.get_mesh_at_time(case_id, float(times[start_index]))
    meshes.append(attach_fields(initial_mesh, current_fields.cpu()))

    squared_error = torch.zeros(len(FIELD_NAMES), dtype=torch.float64)
    value_count = 0
    for offset in range(rollout_steps):
        time_index = start_index + offset
        input_mesh = dataset.get_mesh_at_time(
            case_id, float(times[time_index])
        ).to(device)
        input_mesh = attach_fields(input_mesh, current_fields)
        next_fields = model.predict_next(input_mesh)

        target = truth[time_index + 1].to(device)
        squared_error += (next_fields - target).double().square().sum(dim=0).cpu()
        value_count += target.shape[0]

        result_mesh = dataset.get_mesh_at_time(
            case_id, float(times[time_index + 1])
        )
        meshes.append(attach_fields(result_mesh, next_fields.cpu()))
        current_fields = next_fields

    rmse = torch.sqrt(squared_error / value_count).tolist()
    metrics = {
        f"rollout_rmse_{name}": value for name, value in zip(FIELD_NAMES, rmse)
    }
    metrics["steps"] = rollout_steps
    return meshes, metrics


def parse_args():
    self_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Autoregressive Transolver rollout.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "计算有限元数据" / "comsol_hdf5",
    )
    parser.add_argument("--parameters-json", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=self_root / "rollouts")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    dataset = FpcDataset(
        data_root=str(args.data_root),
        split="all",
        parameters_json=(
            str(args.parameters_json) if args.parameters_json is not None else None
        ),
        field_names=FIELD_NAMES,
    )
    model = load_model(args.checkpoint, device)
    meshes, metrics = rollout_case(
        dataset,
        model,
        args.case_id,
        device,
        start_index=args.start_index,
        steps=args.steps,
    )

    output_path = args.output
    if output_path.suffix != ".pt":
        output_path = output_path / f"{args.case_id}_rollout.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case_id": args.case_id,
            "field_names": FIELD_NAMES,
            "meshes": meshes,
            "metrics": metrics,
        },
        output_path,
    )
    print(f"saved={output_path}")
    print(" ".join(f"{key}={value}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
