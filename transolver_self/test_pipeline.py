import unittest

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from meshGraphNet_self.graph import build_graph_transform
from meshGraphNet_self.train import evaluate, train_one_epoch
from meshGraphNet_self.utils.normalization import Normalizer
from transolver_self.model.simulator import TransolverSimulator, pack_pyg_nodes


class TransolverPipelineTest(unittest.TestCase):
    @staticmethod
    def make_graph(node_count: int, offset: float) -> Data:
        z = torch.linspace(0.0, 1.0, node_count) + offset
        pos = torch.stack([torch.zeros_like(z), z], dim=-1)
        pressure = torch.linspace(1.0, 2.0, node_count)
        temperature = torch.linspace(300.0, 301.0, node_count)
        fields = torch.stack([pressure, temperature], dim=-1)
        face = torch.stack(
            [
                torch.zeros(node_count - 2, dtype=torch.long),
                torch.arange(1, node_count - 1),
                torch.arange(2, node_count),
            ]
        )
        return Data(
            x=torch.cat([torch.zeros(node_count, 1), fields], dim=-1),
            y=fields + torch.tensor([0.1, 0.2]),
            pos=pos,
            face=face,
            mesh_velocity=torch.stack([torch.zeros_like(z), 0.1 * z], dim=-1),
            mesh_region=torch.arange(node_count) % 3,
            case_features=torch.ones(1, 10),
            time=torch.tensor([offset]),
            piston_displacement=torch.tensor([0.01 * offset]),
            piston_velocity=torch.tensor([0.02]),
        )

    def test_variable_node_batch_forward_backward(self):
        graph = next(
            iter(
                DataLoader(
                    [self.make_graph(3, 0.0), self.make_graph(5, 1.0)],
                    batch_size=2,
                )
            )
        )
        model = TransolverSimulator(
            hidden_size=16, layers=2, heads=4, slice_num=4
        )
        self.assertIsInstance(model.field_normalizer, Normalizer)

        dense, mask = pack_pyg_nodes(graph.pos, graph)
        self.assertEqual(dense.shape, (2, 5, 2))
        self.assertEqual(mask.sum(dim=1).tolist(), [3, 5])

        model.train()
        predicted, target = model(graph)
        self.assertEqual(predicted.shape, (8, 2))
        self.assertEqual(target.shape, (8, 2))
        torch.nn.functional.mse_loss(predicted, target).backward()

        model.eval()
        next_fields = model(graph)
        self.assertEqual(next_fields.shape, (8, 2))
        self.assertTrue(torch.isfinite(next_fields).all())

    def test_reused_meshgraphnet_train_and_metrics(self):
        loader = DataLoader(
            [self.make_graph(3, 0.0), self.make_graph(5, 1.0)],
            batch_size=2,
        )
        model = TransolverSimulator(
            hidden_size=16, layers=2, heads=4, slice_num=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        transform = build_graph_transform()

        train_mse = train_one_epoch(
            model,
            loader,
            optimizer,
            transform,
            torch.device("cpu"),
            1.0,
            1,
            1,
        )
        metrics = evaluate(model, loader, transform, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(train_mse)))
        self.assertEqual(
            set(metrics), {"normalized_mse", "rmse_p", "rmse_T"}
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(list(metrics.values())))))


if __name__ == "__main__":
    unittest.main()
