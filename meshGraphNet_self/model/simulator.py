from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .network import EncoderProcessorDecoder

try:
    from ..utils.normalization import Normalizer
except ImportError:
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

    def _graph_context(self, graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        case_features = graph.case_features.reshape(-1, self.case_feature_count)
        time = graph.time.reshape(-1, 1)
        displacement = graph.piston_displacement.reshape(-1, 1)
        velocity = graph.piston_velocity.reshape(-1, 1)
        context = torch.cat([case_features, time, displacement, velocity], dim=-1)

        if hasattr(graph, "batch") and graph.batch is not None:
            graph_index = graph.batch
        else:
            graph_index = torch.zeros(
                graph.num_nodes, dtype=torch.long, device=graph.x.device
            )
        return context, graph_index

    def _build_model_graph(self, graph: Data, accumulate: bool) -> Data:
        current_fields = graph.x[:, 1 : 1 + self.field_count]
        context, graph_index = self._graph_context(graph)

        field_features = self.field_normalizer(current_fields, accumulate)
        position_features = self.position_normalizer(graph.pos, accumulate)
        velocity_features = self.mesh_velocity_normalizer(
            graph.mesh_velocity, accumulate
        )
        context_features = self.context_normalizer(context, accumulate)[graph_index]
        region_features = F.one_hot(
            graph.mesh_region.long(), num_classes=self.region_count
        ).to(current_fields.dtype)

        node_features = torch.cat(
            [
                field_features,
                position_features,
                velocity_features,
                region_features,
                context_features,
            ],
            dim=-1,
        )
        edge_features = self.edge_normalizer(graph.edge_attr, accumulate)
        return Data(
            x=node_features,
            edge_attr=edge_features,
            edge_index=graph.edge_index,
        )

    def normalized_prediction_and_target(
        self, graph: Data, accumulate: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_graph = self._build_model_graph(graph, accumulate)
        predicted_delta = self.network(model_graph)
        current_fields = graph.x[:, 1 : 1 + self.field_count]
        target_delta = graph.y - current_fields
        normalized_target = self.output_normalizer(target_delta, accumulate)
        return predicted_delta, normalized_target

    def predict_next(self, graph: Data) -> torch.Tensor:
        model_graph = self._build_model_graph(graph, accumulate=False)
        normalized_delta = self.network(model_graph)
        current_fields = graph.x[:, 1 : 1 + self.field_count]
        return current_fields + self.output_normalizer.inverse(normalized_delta)

    def forward(self, graph: Data):
        if self.training:
            return self.normalized_prediction_and_target(graph, accumulate=True)
        return self.predict_next(graph)
