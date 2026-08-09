import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Feature normalizer with stable, streamable training-set statistics.

    ``acc_sum`` and ``acc_sum_squared`` retain their historical state-dict names
    for checkpoint compatibility. In statistics version 2 they store the
    running mean and Welford M2 respectively, both in float64.
    """

    STATISTICS_VERSION = 2

    def __init__(
        self,
        size: int,
        max_accumulations: int = 1_000_000,
        std_epsilon: float = 1.0e-8,
    ):
        super().__init__()
        self.size = size
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon
        self.register_buffer("acc_count", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer(
            "num_accumulations", torch.tensor(0.0, dtype=torch.float64)
        )
        self.register_buffer("acc_sum", torch.zeros(1, size, dtype=torch.float64))
        self.register_buffer(
            "acc_sum_squared", torch.zeros(1, size, dtype=torch.float64)
        )
        self.register_buffer(
            "statistics_version",
            torch.tensor(self.STATISTICS_VERSION, dtype=torch.int64),
        )

    def forward(self, data: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        if accumulate and not self.frozen:
            self._accumulate(data.detach())
        mean = self.mean.to(device=data.device, dtype=data.dtype)
        std = self.std.to(device=data.device, dtype=data.dtype)
        return (data - mean) / std

    def inverse(self, normalized_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(
            device=normalized_data.device, dtype=normalized_data.dtype
        )
        std = self.std.to(device=normalized_data.device, dtype=normalized_data.dtype)
        return normalized_data * std + mean

    @property
    def mean(self) -> torch.Tensor:
        return self.acc_sum

    @property
    def raw_std(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        variance = self.acc_sum_squared / safe_count
        return torch.sqrt(torch.clamp(variance, min=0.0))

    @property
    def std(self) -> torch.Tensor:
        raw_std = self.raw_std
        # A truly constant feature normalizes to zero; using one avoids an
        # artificial 1e8 scale while preserving that zero value.
        return torch.where(
            raw_std > self.std_epsilon,
            raw_std,
            torch.ones_like(raw_std),
        )

    @property
    def frozen(self) -> bool:
        return self.num_accumulations.item() >= self.max_accumulations

    @torch.no_grad()
    def reset(self) -> None:
        self.acc_count.zero_()
        self.num_accumulations.zero_()
        self.acc_sum.zero_()
        self.acc_sum_squared.zero_()
        self.statistics_version.fill_(self.STATISTICS_VERSION)

    @torch.no_grad()
    def freeze(self) -> None:
        if self.acc_count.item() <= 0:
            raise RuntimeError("Cannot freeze a normalizer without fitted statistics.")
        self.num_accumulations.fill_(float(self.max_accumulations))

    @torch.no_grad()
    def _accumulate(self, data: torch.Tensor) -> None:
        values = data.reshape(-1, self.size).to(
            device=self.acc_sum.device, dtype=torch.float64
        )
        batch_count = values.shape[0]
        if batch_count == 0:
            return

        batch_mean = values.mean(dim=0, keepdim=True)
        centered = values - batch_mean
        batch_m2 = centered.square().sum(dim=0, keepdim=True)

        old_count = self.acc_count.clone()
        new_count = old_count + batch_count
        if old_count.item() == 0:
            new_mean = batch_mean
            new_m2 = batch_m2
        else:
            delta = batch_mean - self.acc_sum
            new_mean = self.acc_sum + delta * (batch_count / new_count)
            new_m2 = (
                self.acc_sum_squared
                + batch_m2
                + delta.square() * old_count * batch_count / new_count
            )

        self.acc_count.copy_(new_count)
        self.acc_sum.copy_(new_mean)
        self.acc_sum_squared.copy_(new_m2)
        self.num_accumulations.add_(1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version_key = prefix + "statistics_version"
        if version_key not in state_dict:
            count_key = prefix + "acc_count"
            sum_key = prefix + "acc_sum"
            squared_key = prefix + "acc_sum_squared"
            if all(key in state_dict for key in (count_key, sum_key, squared_key)):
                count = state_dict[count_key].to(torch.float64)
                safe_count = torch.clamp(count, min=1.0)
                old_sum = state_dict[sum_key].to(torch.float64)
                old_squared_sum = state_dict[squared_key].to(torch.float64)
                mean = old_sum / safe_count
                m2 = old_squared_sum - old_sum.square() / safe_count
                state_dict[sum_key] = mean
                state_dict[squared_key] = torch.clamp(m2, min=0.0)
                state_dict[count_key] = count
                state_dict[version_key] = torch.tensor(
                    self.STATISTICS_VERSION, dtype=torch.int64
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
