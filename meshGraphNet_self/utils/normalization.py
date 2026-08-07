import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(
        self,
        size: int,
        max_accumulations: int = 1_000_000,
        std_epsilon: float = 1.0e-8,
    ):
        super().__init__()
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon
        self.register_buffer("acc_count", torch.tensor(0.0))
        self.register_buffer("num_accumulations", torch.tensor(0.0))
        self.register_buffer("acc_sum", torch.zeros(1, size))
        self.register_buffer("acc_sum_squared", torch.zeros(1, size))

    def forward(self, data: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        if accumulate and self.num_accumulations.item() < self.max_accumulations:
            self._accumulate(data.detach())
        return (data - self.mean) / self.std

    def inverse(self, normalized_data: torch.Tensor) -> torch.Tensor:
        return normalized_data * self.std + self.mean

    @property
    def mean(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        return self.acc_sum / safe_count

    @property
    def std(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        variance = self.acc_sum_squared / safe_count - self.mean.square()
        return torch.sqrt(torch.clamp(variance, min=self.std_epsilon**2))

    @torch.no_grad()
    def _accumulate(self, data: torch.Tensor) -> None:
        self.acc_sum.add_(data.sum(dim=0, keepdim=True))
        self.acc_sum_squared.add_(data.square().sum(dim=0, keepdim=True))
        self.acc_count.add_(data.shape[0])
        self.num_accumulations.add_(1.0)
