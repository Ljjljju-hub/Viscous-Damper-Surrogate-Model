"""THUML Transolver irregular-mesh architecture without optional dependencies.

Adapted from https://github.com/thuml/Transolver at revision
75e0f67643806a81cd1d3f6adc88dd8c02416fe7 under the MIT license.
"""

from typing import Callable

import torch
import torch.nn as nn

from .physics_attention import PhysicsAttentionIrregularMesh


ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        layers: int = 1,
        activation: str = "gelu",
        residual: bool = True,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        act = ACTIVATIONS[activation]
        self.residual = residual
        self.pre = nn.Sequential(nn.Linear(input_size, hidden_size), act())
        self.hidden = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_size, hidden_size), act())
            for _ in range(layers)
        )
        self.post = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        for layer in self.hidden:
            update = layer(x)
            x = x + update if self.residual else update
        return self.post(x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        hidden_size: int,
        dropout: float,
        activation: str,
        mlp_ratio: int,
        last_layer: bool,
        output_size: int,
        slice_num: int,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = PhysicsAttentionIrregularMesh(
            hidden_size,
            heads=heads,
            dim_head=hidden_size // heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(
            hidden_size,
            hidden_size * mlp_ratio,
            hidden_size,
            layers=0,
            activation=activation,
            residual=False,
        )
        if last_layer:
            self.norm3 = nn.LayerNorm(hidden_size)
            self.output = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, node_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.attention(self.norm1(x), node_mask=node_mask) + x
        x = self.mlp(self.norm2(x)) + x
        if self.last_layer:
            x = self.output(self.norm3(x))
            if node_mask is not None:
                x = x * node_mask[..., None].to(x.dtype)
        return x


class Transolver(nn.Module):
    """Official irregular-mesh Transolver interface: ``forward(x, fx)``."""

    def __init__(
        self,
        *,
        space_dim: int = 2,
        function_dim: int = 20,
        output_size: int = 2,
        layers: int = 8,
        hidden_size: int = 256,
        heads: int = 8,
        slice_num: int = 32,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        activation: str = "gelu",
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")

        self.preprocess = MLP(
            space_dim + function_dim,
            hidden_size * 2,
            hidden_size,
            layers=0,
            activation=activation,
            residual=False,
        )
        self.blocks = nn.ModuleList(
            TransolverBlock(
                heads=heads,
                hidden_size=hidden_size,
                dropout=dropout,
                activation=activation,
                mlp_ratio=mlp_ratio,
                last_layer=index == layers - 1,
                output_size=output_size,
                slice_num=slice_num,
            )
            for index in range(layers)
        )
        self.placeholder = nn.Parameter(
            torch.rand(hidden_size, dtype=torch.float32) / hidden_size
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        position: torch.Tensor,
        function: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.preprocess(torch.cat([position, function], dim=-1))
        x = x + self.placeholder[None, None, :]
        if node_mask is not None:
            x = x * node_mask[..., None].to(x.dtype)
        for block in self.blocks:
            x = block(x, node_mask=node_mask)
        return x
