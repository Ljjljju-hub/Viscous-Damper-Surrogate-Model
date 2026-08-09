from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch_geometric.data import Data

from meshGraphNet_self.dataset import FpcDataset
from meshGraphNet_self.experiment_utils import load_split_manifest, verify_manifest_snapshot
from meshGraphNet_self.graph import build_graph_transform, prepare_graph
from meshGraphNet_self.model.simulator import SurrogateSimulator
from meshGraphNet_self.training import choose_device
from meshGraphNet_self.utils.utils import NodeType
from transolver_self.model.simulator import TransolverSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    output_mean: torch.Tensor
    output_std: torch.Tensor


@dataclass
class EvaluationContext:
    train_size: int
    seed: int
    model_names: tuple[str, ...]
    device: torch.device
    manifest: dict
    dataset: FpcDataset
    checkpoints: dict[str, Path]
    checkpoint_hashes: dict[str, str]
    graph_transform: object
    _models: dict[str, LoadedModel] = field(default_factory=dict)

    def get_model(self, model_name: str) -> LoadedModel:
        if model_name not in self.model_names:
            raise KeyError(f"Model {model_name!r} is not configured.")
        if model_name not in self._models:
            self._models[model_name] = load_model(
                model_name, self.checkpoints[model_name], self.device
            )
        return self._models[model_name]


def checkpoint_path(model_name: str, train_size: int, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "training_workspace"
        / "runs"
        / model_name
        / f"n{train_size:04d}"
        / f"seed_{seed}"
        / "checkpoints"
        / "best.pt"
    )


def load_model(
    model_name: str, checkpoint: Path, device: torch.device
) -> LoadedModel:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("model_name") != model_name:
        raise ValueError(
            f"Checkpoint model_name={state.get('model_name')!r}, expected {model_name!r}."
        )
    factories = {
        "meshgraphnet": SurrogateSimulator,
        "transolver": TransolverSimulator,
    }
    if model_name not in factories:
        raise ValueError(f"Unknown model: {model_name}")
    model = factories[model_name](**state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return LoadedModel(
        name=model_name,
        model=model,
        checkpoint_path=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        output_mean=model.output_normalizer.mean.detach().cpu().reshape(-1),
        output_std=model.output_normalizer.std.detach().cpu().reshape(-1),
    )


def load_evaluation_context(
    *,
    models: list[str] | tuple[str, ...],
    train_size: int,
    seed: int,
    device: str,
    manifest_path: Path | None = None,
) -> EvaluationContext:
    manifest_path = manifest_path or (
        PROJECT_ROOT / "training_workspace" / "dataset_split" / "split_manifest.json"
    )
    manifest = load_split_manifest(manifest_path)
    snapshot_errors = verify_manifest_snapshot(manifest)
    if snapshot_errors:
        raise RuntimeError("Frozen dataset snapshot changed:\n" + "\n".join(snapshot_errors[:20]))
    return _build_evaluation_context(
        models=models,
        train_size=train_size,
        seed=seed,
        device=device,
        data_root=Path(manifest["data_root"]),
        parameters_json=Path(manifest["parameters_json"]),
        case_ids=list(manifest["test"]),
        manifest=manifest,
    )


def load_evaluation_context_from_cases(
    *,
    models: list[str] | tuple[str, ...],
    train_size: int,
    seed: int,
    device: str,
    data_root: Path,
    parameters_json: Path,
    case_ids: list[str] | tuple[str, ...],
    source_name: str,
) -> EvaluationContext:
    data_root = Path(data_root).resolve()
    parameters_json = Path(parameters_json).resolve()
    selected_cases = list(case_ids)
    manifest = {
        "source_name": str(source_name),
        "data_root": str(data_root),
        "parameters_json": str(parameters_json),
        "test": selected_cases,
    }
    return _build_evaluation_context(
        models=models,
        train_size=train_size,
        seed=seed,
        device=device,
        data_root=data_root,
        parameters_json=parameters_json,
        case_ids=selected_cases,
        manifest=manifest,
    )


def _build_evaluation_context(
    *,
    models: list[str] | tuple[str, ...],
    train_size: int,
    seed: int,
    device: str,
    data_root: Path,
    parameters_json: Path,
    case_ids: list[str] | tuple[str, ...],
    manifest: dict,
) -> EvaluationContext:
    model_names = tuple(models)
    unknown = [name for name in model_names if name not in {"meshgraphnet", "transolver"}]
    if not model_names:
        raise ValueError("At least one evaluation model is required.")
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    selected_cases = list(case_ids)
    if not selected_cases:
        raise ValueError("At least one evaluation case is required.")
    if len(selected_cases) != len(set(selected_cases)):
        raise ValueError("Evaluation case_ids contains duplicates.")
    data_root = Path(data_root).resolve()
    parameters_json = Path(parameters_json).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Evaluation data root does not exist: {data_root}")
    if not parameters_json.is_file():
        raise FileNotFoundError(
            f"Evaluation parameters JSON does not exist: {parameters_json}"
        )
    dataset = FpcDataset(
        data_root=str(data_root),
        split="test",
        parameters_json=str(parameters_json),
        case_ids=selected_cases,
    )
    checkpoints = {
        name: checkpoint_path(name, train_size, seed) for name in model_names
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    return EvaluationContext(
        train_size=train_size,
        seed=seed,
        model_names=model_names,
        device=choose_device(device),
        manifest=manifest,
        dataset=dataset,
        checkpoints=checkpoints,
        checkpoint_hashes={name: file_sha256(path) for name, path in checkpoints.items()},
        graph_transform=build_graph_transform(),
    )


def attach_fields(mesh: Data, fields: torch.Tensor) -> Data:
    node_type = torch.full(
        (fields.shape[0], 1),
        int(NodeType.NORMAL),
        dtype=fields.dtype,
        device=fields.device,
    )
    mesh.x = torch.cat([node_type, fields], dim=-1)
    return mesh


@torch.no_grad()
def predict_next(
    model_name: str,
    loaded_model: LoadedModel,
    graph: Data,
    graph_transform,
) -> torch.Tensor:
    graph = graph.to(next(loaded_model.model.parameters()).device)
    if model_name == "meshgraphnet":
        graph = prepare_graph(graph, graph_transform)
    return loaded_model.model.predict_next(graph)
