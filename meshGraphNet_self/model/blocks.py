import torch
import torch.nn as nn
from torch_geometric.data import Data


class EdgeBlock(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, graph: Data) -> Data:
        senders, receivers = graph.edge_index
        edge_inputs = torch.cat(
            [graph.x[senders], graph.x[receivers], graph.edge_attr], dim=-1
        )
        edge_attr = self.network(edge_inputs)
        return Data(x=graph.x, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, graph: Data) -> Data:
        _, receivers = graph.edge_index
        aggregated = graph.edge_attr.new_zeros(
            (graph.num_nodes, graph.edge_attr.shape[-1])
        )
        aggregated.index_add_(0, receivers, graph.edge_attr)
        node_inputs = torch.cat([graph.x, aggregated], dim=-1)
        x = self.network(node_inputs)
        return Data(x=x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)
