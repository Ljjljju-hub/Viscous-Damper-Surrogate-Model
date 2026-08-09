import torch.nn as nn
from torch_geometric.data import Data

from .blocks import EdgeBlock, NodeBlock


def build_mlp(
    input_size: int,
    hidden_size: int,
    output_size: int,
    layer_norm: bool = True,
) -> nn.Module:
    layers = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    if layer_norm:
        return nn.Sequential(layers, nn.LayerNorm(output_size))
    return layers


class Encoder(nn.Module):
    def __init__(self, node_input_size: int, edge_input_size: int, hidden_size: int):
        super().__init__()
        self.node_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
        self.edge_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)

    def forward(self, graph: Data) -> Data:
        return Data(
            x=self.node_encoder(graph.x),
            edge_attr=self.edge_encoder(graph.edge_attr),
            edge_index=graph.edge_index,
        )


class GraphNetworkBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.edge_block = EdgeBlock(
            build_mlp(3 * hidden_size, hidden_size, hidden_size)
        )
        self.node_block = NodeBlock(
            build_mlp(2 * hidden_size, hidden_size, hidden_size)
        )

    def forward(self, graph: Data) -> Data:
        previous_x = graph.x
        previous_edge_attr = graph.edge_attr
        graph = self.edge_block(graph)
        graph = self.node_block(graph)
        return Data(
            x=previous_x + graph.x,
            edge_attr=previous_edge_attr + graph.edge_attr,
            edge_index=graph.edge_index,
        )


class EncoderProcessorDecoder(nn.Module):
    def __init__(
        self,
        message_passing_steps: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.encoder = Encoder(node_input_size, edge_input_size, hidden_size)
        self.processor = nn.ModuleList(
            GraphNetworkBlock(hidden_size) for _ in range(message_passing_steps)
        )
        self.decoder = build_mlp(
            hidden_size, hidden_size, output_size, layer_norm=False
        )

    def forward(self, graph: Data):
        graph = self.encoder(graph)
        for block in self.processor:
            graph = block(graph)
        return self.decoder(graph.x)
