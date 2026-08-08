import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.dataset import FpcDataset
from meshGraphNet_self.experiment_utils import atomic_write_json, load_split_manifest
from meshGraphNet_self.graph import build_graph_transform, prepare_graph
from meshGraphNet_self.model.simulator import SurrogateSimulator
from meshGraphNet_self.training import FIELD_NAMES, choose_device
from transolver_self.model.simulator import TransolverSimulator
from transolver_self.rollout import attach_fields


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_name = checkpoint.get("model_name")
    if model_name == "meshgraphnet":
        model = SurrogateSimulator(**checkpoint["model_config"])
    elif model_name == "transolver":
        model = TransolverSimulator(**checkpoint["model_config"])
    else:
        raise ValueError(f"Unsupported checkpoint model_name={model_name!r}.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model_name, model


@torch.no_grad()
def rollout_one_case(
    dataset: FpcDataset,
    model_name: str,
    model,
    case_id: str,
    device: torch.device,
    save_predictions: bool,
):
    file_path = dataset._resolve_file(case_id)
    with h5py.File(file_path, "r") as h5_file:
        times = torch.as_tensor(h5_file["time_steps"][:], dtype=torch.float32)
        truth = torch.stack(
            [
                torch.as_tensor(h5_file[f"fields/{name}"][:], dtype=torch.float32)
                for name in FIELD_NAMES
            ],
            dim=-1,
        )

    transform = build_graph_transform() if model_name == "meshgraphnet" else None
    current_fields = truth[0].to(device)
    squared_error = torch.zeros(len(FIELD_NAMES), dtype=torch.float64)
    value_count = 0
    meshes = []
    if save_predictions:
        initial = dataset.get_mesh_at_time(case_id, float(times[0]))
        meshes.append(attach_fields(initial, current_fields.cpu()))

    for time_index in range(len(times) - 1):
        input_mesh = dataset.get_mesh_at_time(
            case_id, float(times[time_index])
        )
        input_mesh = attach_fields(input_mesh, current_fields.cpu())
        if transform is not None:
            input_mesh = prepare_graph(input_mesh, transform)
        input_mesh = input_mesh.to(device)
        next_fields = model.predict_next(input_mesh)

        target = truth[time_index + 1].to(device)
        squared_error += (next_fields - target).double().square().sum(dim=0).cpu()
        value_count += target.shape[0]
        current_fields = next_fields

        if save_predictions:
            result = dataset.get_mesh_at_time(
                case_id, float(times[time_index + 1])
            )
            meshes.append(attach_fields(result, next_fields.cpu()))

    rmse = torch.sqrt(squared_error / value_count).tolist()
    metrics = {
        "steps": len(times) - 1,
        "value_count": value_count,
        "squared_error_p": squared_error[0].item(),
        "squared_error_T": squared_error[1].item(),
        "rmse_p": rmse[0],
        "rmse_T": rmse[1],
    }
    return metrics, meshes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one experiment with resumable test-case rollouts."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rollout-count", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    evaluation_path = run_dir / "evaluation.json"
    predictions_dir = run_dir / "rollouts"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = load_split_manifest(Path(summary["split_manifest"]))
    selected_ids = list(manifest["test"])[
        : min(args.rollout_count, len(manifest["test"]))
    ]
    if not selected_ids:
        raise ValueError("rollout-count must select at least one test case.")

    existing = {}
    if evaluation_path.exists() and not args.force:
        existing = json.loads(evaluation_path.read_text(encoding="utf-8"))
        predictions_complete = all(
            (predictions_dir / f"{case_id}.pt").exists()
            for case_id in selected_ids
        )
        if (
            existing.get("completed")
            and existing.get("case_ids") == selected_ids
            and (not args.save_predictions or predictions_complete)
        ):
            print(f"SKIP complete: {evaluation_path}")
            return
    previous_metrics = existing.get("case_metrics", {}) if not args.force else {}
    case_metrics = {
        case_id: previous_metrics[case_id]
        for case_id in selected_ids
        if case_id in previous_metrics
    }

    device = choose_device(args.device)
    model_name, model = load_model(Path(summary["best_checkpoint"]), device)
    dataset = FpcDataset(
        data_root=manifest["data_root"],
        parameters_json=manifest["parameters_json"],
        split="test",
        case_ids=selected_ids,
        field_names=FIELD_NAMES,
    )
    for index, case_id in enumerate(selected_ids, start=1):
        prediction_exists = (predictions_dir / f"{case_id}.pt").exists()
        if case_id in case_metrics and (
            not args.save_predictions or prediction_exists
        ):
            print(f"SKIP rollout {case_id} ({index}/{len(selected_ids)})")
            continue
        print(f"ROLL rollout {case_id} ({index}/{len(selected_ids)})")
        metrics, meshes = rollout_one_case(
            dataset,
            model_name,
            model,
            case_id,
            device,
            args.save_predictions,
        )
        case_metrics[case_id] = metrics
        if args.save_predictions:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "case_id": case_id,
                    "field_names": FIELD_NAMES,
                    "meshes": meshes,
                    "metrics": metrics,
                },
                predictions_dir / f"{case_id}.pt",
            )
        atomic_write_json(
            evaluation_path,
            {
                "completed": False,
                "model_name": model_name,
                "case_ids": selected_ids,
                "case_metrics": case_metrics,
            },
        )

    total_values = sum(item["value_count"] for item in case_metrics.values())
    total_p = sum(item["squared_error_p"] for item in case_metrics.values())
    total_T = sum(item["squared_error_T"] for item in case_metrics.values())
    evaluation = {
        "completed": True,
        "model_name": model_name,
        "case_ids": selected_ids,
        "rollout_case_count": len(selected_ids),
        "rollout_rmse_p": math.sqrt(total_p / total_values),
        "rollout_rmse_T": math.sqrt(total_T / total_values),
        "one_step_test_normalized_mse": summary.get("test_normalized_mse"),
        "one_step_test_rmse_p": summary.get("test_rmse_p"),
        "one_step_test_rmse_T": summary.get("test_rmse_T"),
        "case_metrics": case_metrics,
    }
    atomic_write_json(evaluation_path, evaluation)
    print(f"saved={evaluation_path}")
    print(
        f"rollout_rmse_p={evaluation['rollout_rmse_p']:.6e} "
        f"rollout_rmse_T={evaluation['rollout_rmse_T']:.6e}"
    )


if __name__ == "__main__":
    main()
