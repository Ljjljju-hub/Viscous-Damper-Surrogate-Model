from typing import NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class NormalizedGraphFeatures(NamedTuple):
    fields: torch.Tensor
    position: torch.Tensor
    mesh_velocity: torch.Tensor
    region: torch.Tensor
    context: torch.Tensor

    @property
    def combined(self) -> torch.Tensor:
        return torch.cat(self, dim=-1)


def current_fields(graph: Data, field_count: int) -> torch.Tensor:
    return graph.x[:, 1 : 1 + field_count]


def graph_context(
    graph: Data, case_feature_count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    case_features = graph.case_features.reshape(-1, case_feature_count)
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


def normalize_graph_features(
    graph: Data,
    *,
    field_count: int,
    case_feature_count: int,
    region_count: int,
    field_normalizer,
    position_normalizer,
    mesh_velocity_normalizer,
    context_normalizer,
    accumulate: bool,
) -> NormalizedGraphFeatures:
    """Build the shared MeshGraphNet/Transolver node-feature contract."""
    fields = current_fields(graph, field_count)
    context, graph_index = graph_context(graph, case_feature_count)
    region = F.one_hot(
        graph.mesh_region.long(), num_classes=region_count
    ).to(fields.dtype)
    return NormalizedGraphFeatures(
        fields=field_normalizer(fields, accumulate),
        position=position_normalizer(graph.pos, accumulate),
        mesh_velocity=mesh_velocity_normalizer(
            graph.mesh_velocity, accumulate
        ),
        region=region,
        context=context_normalizer(context, accumulate)[graph_index],
    )


def target_field_delta(graph: Data, field_count: int) -> torch.Tensor:
    return graph.y - current_fields(graph, field_count)
