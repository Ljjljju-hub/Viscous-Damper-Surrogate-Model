from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from meshGraphNet_self.features import (
    current_fields,
    normalize_graph_features,
    target_field_delta,
)
from meshGraphNet_self.utils.normalization import Normalizer

from .transolver import Transolver


def pack_pyg_nodes(
    values: torch.Tensor, graph: Data
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack PyG's concatenated node storage into padded ``[B, N, C]``."""
    if hasattr(graph, "ptr") and graph.ptr is not None:
        ptr = graph.ptr
    else:
        ptr = torch.tensor(
            [0, values.shape[0]], dtype=torch.long, device=values.device
        )
    lengths = ptr[1:] - ptr[:-1]
    batch_size = int(lengths.numel())
    max_nodes = int(lengths.max().item())
    dense = values.new_zeros((batch_size, max_nodes, values.shape[-1]))
    mask = torch.zeros(
        (batch_size, max_nodes), dtype=torch.bool, device=values.device
    )
    for batch_index, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
        count = int((end - start).item())
        dense[batch_index, :count] = values[int(start.item()) : int(end.item())]
        mask[batch_index, :count] = True
    return dense, mask


class TransolverSimulator(nn.Module):
    """MeshGraphNet-compatible wrapper around THUML Transolver."""

    def __init__(
        self,
        field_count: int = 2,
        case_feature_count: int = 10,
        region_count: int = 3,
        layers: int = 8,
        hidden_size: int = 256,
        heads: int = 8,
        slice_num: int = 32,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
    ):
        super().__init__()
        self.field_count = field_count
        self.case_feature_count = case_feature_count
        self.region_count = region_count
        self.context_size = case_feature_count + 3

        # These names and definitions intentionally match SurrogateSimulator.
        self.field_normalizer = Normalizer(field_count)
        self.position_normalizer = Normalizer(2)
        self.mesh_velocity_normalizer = Normalizer(2)
        self.context_normalizer = Normalizer(self.context_size)
        self.output_normalizer = Normalizer(field_count)

        function_dim = (
            field_count + 2 + region_count + self.context_size
        )
        self.network = Transolver(
            space_dim=2,
            function_dim=function_dim,
            output_size=field_count,
            layers=layers,
            hidden_size=hidden_size,
            heads=heads,
            slice_num=slice_num,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )

    def _build_dense_inputs(
        self, graph: Data, accumulate: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = normalize_graph_features(
            graph,
            field_count=self.field_count,
            case_feature_count=self.case_feature_count,
            region_count=self.region_count,
            field_normalizer=self.field_normalizer,
            position_normalizer=self.position_normalizer,
            mesh_velocity_normalizer=self.mesh_velocity_normalizer,
            context_normalizer=self.context_normalizer,
            accumulate=accumulate,
        )
        function = torch.cat(
            [
                features.fields,
                features.mesh_velocity,
                features.region,
                features.context,
            ],
            dim=-1,
        )
        position_dense, mask = pack_pyg_nodes(features.position, graph)
        function_dense, function_mask = pack_pyg_nodes(function, graph)
        if not torch.equal(mask, function_mask):
            raise RuntimeError("Position and function batch layouts do not match.")
        return position_dense, function_dense, mask

    def normalized_prediction_and_target(
        self, graph: Data, accumulate: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position, function, mask = self._build_dense_inputs(graph, accumulate)
        dense_prediction = self.network(position, function, node_mask=mask)
        predicted_delta = dense_prediction[mask]
        target_delta = target_field_delta(graph, self.field_count)
        normalized_target = self.output_normalizer(target_delta, accumulate)
        return predicted_delta, normalized_target

    def normalizers(self):
        return {
            "field": self.field_normalizer,
            "position": self.position_normalizer,
            "mesh_velocity": self.mesh_velocity_normalizer,
            "context": self.context_normalizer,
            "output": self.output_normalizer,
        }

    @torch.no_grad()
    def reset_normalizers(self) -> None:
        for normalizer in self.normalizers().values():
            normalizer.reset()

    @torch.no_grad()
    def accumulate_normalizers(self, graph: Data) -> None:
        self._build_dense_inputs(graph, accumulate=True)
        self.output_normalizer(
            target_field_delta(graph, self.field_count), accumulate=True
        )

    @torch.no_grad()
    def freeze_normalizers(self) -> None:
        for normalizer in self.normalizers().values():
            normalizer.freeze()

    def predict_next(self, graph: Data) -> torch.Tensor:
        position, function, mask = self._build_dense_inputs(
            graph, accumulate=False
        )
        normalized_delta = self.network(position, function, node_mask=mask)[mask]
        return current_fields(
            graph, self.field_count
        ) + self.output_normalizer.inverse(normalized_delta)

    def forward(self, graph: Data):
        if self.training:
            return self.normalized_prediction_and_target(graph, accumulate=False)
        return self.predict_next(graph)
