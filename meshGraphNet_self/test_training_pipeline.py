import importlib.util
import unittest

import torch


PYG_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None

if PYG_AVAILABLE:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from graph import build_graph_transform, prepare_graph
    from model.simulator import SurrogateSimulator


@unittest.skipUnless(PYG_AVAILABLE, "torch_geometric is not installed")
class TrainingPipelineTest(unittest.TestCase):
    @staticmethod
    def make_graph(offset: float) -> "Data":
        pos = torch.tensor(
            [[0.0, offset], [1.0, offset], [0.0, 1.0 + offset]],
            dtype=torch.float32,
        )
        current = torch.tensor(
            [[0.0, 1.0, 300.0], [0.0, 2.0, 301.0], [0.0, 3.0, 302.0]]
        )
        target = current[:, 1:] + torch.tensor([0.1, 0.2])
        return Data(
            x=current,
            y=target,
            pos=pos,
            face=torch.tensor([[0], [1], [2]], dtype=torch.long),
            mesh_velocity=torch.zeros(3, 2),
            mesh_region=torch.tensor([0, 1, 2]),
            case_features=torch.ones(1, 10),
            time=torch.tensor([0.0]),
            piston_displacement=torch.tensor([0.0]),
            piston_velocity=torch.tensor([0.0]),
        )

    def test_batch_forward_backward_and_inference(self):
        loader = DataLoader(
            [self.make_graph(0.0), self.make_graph(2.0)], batch_size=2
        )
        graph = prepare_graph(next(iter(loader)), build_graph_transform())
        model = SurrogateSimulator(hidden_size=16, message_passing_steps=2)

        model.train()
        predicted, target = model(graph)
        self.assertEqual(predicted.shape, (6, 2))
        torch.nn.functional.mse_loss(predicted, target).backward()

        model.eval()
        prediction = model(graph)
        self.assertEqual(prediction.shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
