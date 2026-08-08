from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .network import EncoderProcessorDecoder

try:
    from ..features import (
        current_fields,
        normalize_graph_features,
        target_field_delta,
    )
    from ..utils.normalization import Normalizer
except ImportError:
    from features import current_fields, normalize_graph_features, target_field_delta
    from utils.normalization import Normalizer


class SurrogateSimulator(nn.Module):
    def __init__(
        self,
        field_count: int = 2,
        case_feature_count: int = 10,
        region_count: int = 3,
        edge_input_size: int = 3,
        hidden_size: int = 128,
        message_passing_steps: int = 15,
    ):
        super().__init__()
        self.field_count = field_count
        self.case_feature_count = case_feature_count
        self.region_count = region_count
        self.context_size = case_feature_count + 3

        self.field_normalizer = Normalizer(field_count)
        self.position_normalizer = Normalizer(2)
        self.mesh_velocity_normalizer = Normalizer(2)
        self.context_normalizer = Normalizer(self.context_size)
        self.edge_normalizer = Normalizer(edge_input_size)
        self.output_normalizer = Normalizer(field_count)

        node_input_size = field_count + 2 + 2 + region_count + self.context_size
        self.network = EncoderProcessorDecoder(
            message_passing_steps=message_passing_steps,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            output_size=field_count,
            hidden_size=hidden_size,
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _build_model_graph(self, graph: Data, accumulate: bool) -> Data:
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
        edge_features = self.edge_normalizer(graph.edge_attr, accumulate)
        return Data(
            x=features.combined,
            edge_attr=edge_features,
            edge_index=graph.edge_index,
        )

    def normalized_prediction_and_target(
        self, graph: Data, accumulate: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_graph = self._build_model_graph(graph, accumulate)
        predicted_delta = self.network(model_graph)
        target_delta = target_field_delta(graph, self.field_count)
        normalized_target = self.output_normalizer(target_delta, accumulate)
        return predicted_delta, normalized_target

    def normalizers(self):
        return {
            "field": self.field_normalizer,
            "position": self.position_normalizer,
            "mesh_velocity": self.mesh_velocity_normalizer,
            "context": self.context_normalizer,
            "edge": self.edge_normalizer,
            "output": self.output_normalizer,
        }

    @torch.no_grad()
    def reset_normalizers(self) -> None:
        for normalizer in self.normalizers().values():
            normalizer.reset()

    @torch.no_grad()
    def accumulate_normalizers(self, graph: Data) -> None:
        self._build_model_graph(graph, accumulate=True)
        self.output_normalizer(
            target_field_delta(graph, self.field_count), accumulate=True
        )

    @torch.no_grad()
    def freeze_normalizers(self) -> None:
        for normalizer in self.normalizers().values():
            normalizer.freeze()

    def predict_next(self, graph: Data) -> torch.Tensor:
        model_graph = self._build_model_graph(graph, accumulate=False)
        normalized_delta = self.network(model_graph)
        return current_fields(
            graph, self.field_count
        ) + self.output_normalizer.inverse(normalized_delta)

    def forward(self, graph: Data):
        if self.training:
            return self.normalized_prediction_and_target(graph, accumulate=False)
        return self.predict_next(graph)
