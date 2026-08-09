import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

try:
    from .mesh_motion import DamperMeshMotion
    from .utils.utils import NodeType
except ImportError:
    from mesh_motion import DamperMeshMotion
    from utils.utils import NodeType


CASE_FEATURE_NAMES = (
    "c",
    "sx",
    "sy",
    "r1",
    "a2",
    "b1",
    "b2",
    "A",
    "Ts",
    "mu",
)


def load_case_parameters(json_path: Path) -> Dict[str, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {item["case_id"]: item for item in raw["parameters_list"]}


def case_id_from_path(path: Path) -> str:
    return path.stem


def build_case_features(params: dict, unit_scale: float) -> torch.Tensor:
    geometry = params["geometry"]
    loading = params["loading"]
    material = params["material"]
    values = [
        float(geometry["c"]) * unit_scale,
        float(geometry["sx"]) * unit_scale,
        float(geometry["sy"]) * unit_scale,
        float(geometry["r1"]) * unit_scale,
        float(geometry["a2"]) * unit_scale,
        float(geometry["b1"]) * unit_scale,
        float(geometry["b2"]) * unit_scale,
        float(loading["A"]) * unit_scale,
        float(loading["Ts"]),
        float(material["mu"]),
    ]
    return torch.tensor([values], dtype=torch.float32)


def find_default_parameters_json(data_root: Path) -> Optional[Path]:
    candidates = [
        data_root / "4_Combined_Master_Dataset.json",
        data_root.parent / "4_Combined_Master_Dataset.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def vtk_cells_to_tri_faces(cells: np.ndarray) -> np.ndarray:
    """Convert flattened VTK cells to triangular faces for PyG FaceToEdge."""
    faces = []
    i = 0
    cells = np.asarray(cells, dtype=np.int64).reshape(-1)
    while i < len(cells):
        n = int(cells[i])
        node_ids = cells[i + 1 : i + 1 + n]
        if len(node_ids) != n:
            raise ValueError("Truncated cell in mesh/connectivity.")
        if n == 3:
            faces.append(node_ids)
        elif n > 3:
            for j in range(1, n - 1):
                faces.append([node_ids[0], node_ids[j], node_ids[j + 1]])
        i += n + 1
    if not faces:
        raise ValueError("No valid cells found in mesh/connectivity.")
    return np.asarray(faces, dtype=np.int64)


class FpcDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        parameters_json: Optional[str] = None,
        unit_scale: float = 1.0e-3,
        validate_mesh_domain: bool = True,
        field_names: Iterable[str] = ("p", "T"),
        case_ids: Optional[Sequence[str]] = None,
    ):
        super().__init__(data_root, transform=None, pre_transform=None)

        self.data_root = Path(data_root).resolve()
        self.split = split
        self.unit_scale = unit_scale
        self.validate_mesh_domain = validate_mesh_domain
        self.field_names = tuple(field_names)

        json_path = (
            Path(parameters_json).resolve()
            if parameters_json is not None
            else find_default_parameters_json(self.data_root)
        )
        if json_path is None:
            raise FileNotFoundError(
                "Could not find 4_Combined_Master_Dataset.json. "
                "Pass parameters_json explicitly."
            )
        self.case_parameters = load_case_parameters(json_path)

        all_files = sorted(self.data_root.glob("*.h5"))
        if not all_files:
            raise FileNotFoundError(f"No .h5 files found under {self.data_root}.")
        self.file_by_case_id = {case_id_from_path(path): path for path in all_files}

        if case_ids is not None:
            requested = list(case_ids)
            if len(requested) != len(set(requested)):
                raise ValueError("case_ids contains duplicate case identifiers.")
            missing = [
                case_id for case_id in requested if case_id not in self.file_by_case_id
            ]
            if missing:
                preview = ", ".join(missing[:10])
                raise FileNotFoundError(
                    f"{len(missing)} requested cases are missing under "
                    f"{self.data_root}: {preview}"
                )
            self.files = [self.file_by_case_id[case_id] for case_id in requested]
        else:
            num_files = len(all_files)
            train_idx = int(num_files * 0.8)
            valid_idx = int(num_files * 0.9)
            if split == "train":
                self.files = all_files[:train_idx]
            elif split in ("valid", "val"):
                self.files = all_files[train_idx:valid_idx]
            elif split == "test":
                self.files = all_files[valid_idx:]
            elif split == "all":
                self.files = all_files
            else:
                raise ValueError(
                    "split must be one of: train, valid, val, test, all."
                )

        self.index_map = []
        self.mesh_cache = {}
        for file_path in self.files:
            case_id = case_id_from_path(file_path)
            if case_id not in self.case_parameters:
                raise KeyError(f"{case_id} is missing from {json_path}.")
            with h5py.File(str(file_path), "r") as f:
                num_steps = len(f["time_steps"])
            for time_index in range(num_steps - 1):
                self.index_map.append((file_path, time_index))

        print(f"[{split.upper()}] loaded {len(self.index_map)} graph samples.")

    def len(self):
        return len(self.index_map)

    def _resolve_file(self, case: Union[str, Path]) -> Path:
        path = Path(case)
        if path.suffix == ".h5":
            return path.resolve()
        case_id = str(case)
        if case_id not in self.file_by_case_id:
            raise KeyError(f"Unknown case_id={case_id!r}.")
        return self.file_by_case_id[case_id]

    def _load_mesh_cache(self, file_path: Path) -> dict:
        if file_path in self.mesh_cache:
            return self.mesh_cache[file_path]

        case_id = case_id_from_path(file_path)
        params = self.case_parameters[case_id]
        with h5py.File(str(file_path), "r") as f:
            reference_pos = f["mesh/coordinates"][:, :2].astype(np.float64)
            faces = vtk_cells_to_tri_faces(f["mesh/connectivity"][:])

        motion = DamperMeshMotion(
            reference_pos=reference_pos,
            geometry=params["geometry"],
            loading=params["loading"],
            unit_scale=self.unit_scale,
            validate_domain=self.validate_mesh_domain,
        )
        cached = {
            "face": torch.as_tensor(faces.T.copy(), dtype=torch.long),
            "motion": motion,
            "params": params,
        }
        self.mesh_cache[file_path] = cached
        return cached

    def get_mesh_at_time(self, case: Union[str, Path], time: float) -> Data:
        """Return the restored PyG mesh object for an arbitrary physical time."""
        file_path = self._resolve_file(case)
        mesh = self._load_mesh_cache(file_path)
        state = mesh["motion"].at_time(float(time))
        return Data(
            pos=torch.as_tensor(state.pos, dtype=torch.float32),
            face=mesh["face"],
            mesh_region=torch.as_tensor(state.region, dtype=torch.long),
            mesh_motion_weight=torch.as_tensor(
                state.motion_weight, dtype=torch.float32
            ),
            node_displacement=torch.as_tensor(
                state.node_displacement, dtype=torch.float32
            ).unsqueeze(1),
            piston_displacement=torch.tensor(
                [state.piston_displacement], dtype=torch.float32
            ),
            mesh_velocity=torch.as_tensor(
                state.mesh_velocity, dtype=torch.float32
            ),
            piston_velocity=torch.tensor(
                [state.piston_velocity], dtype=torch.float32
            ),
            case_features=build_case_features(mesh["params"], self.unit_scale),
            time=torch.tensor([state.time], dtype=torch.float32),
            case_id=case_id_from_path(file_path),
        )

    def get(self, idx):
        file_path, time_index = self.index_map[idx]
        with h5py.File(str(file_path), "r") as f:
            time_value = float(f["time_steps"][time_index])
            current_fields = [
                torch.as_tensor(
                    f[f"fields/{name}"][time_index, :], dtype=torch.float32
                ).unsqueeze(1)
                for name in self.field_names
            ]
            next_fields = [
                torch.as_tensor(
                    f[f"fields/{name}"][time_index + 1, :], dtype=torch.float32
                ).unsqueeze(1)
                for name in self.field_names
            ]

        graph = self.get_mesh_at_time(file_path, time_value)
        node_type = torch.full(
            (graph.pos.shape[0], 1),
            int(NodeType.NORMAL),
            dtype=torch.float32,
        )
        graph.x = torch.cat([node_type, *current_fields], dim=-1)
        graph.y = torch.cat(next_fields, dim=-1)
        return graph
