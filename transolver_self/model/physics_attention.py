"""Physics-Attention adapted from THUML Transolver's irregular-mesh model.

Upstream: https://github.com/thuml/Transolver
Revision: 75e0f67643806a81cd1d3f6adc88dd8c02416fe7
License: MIT; see transolver_self/THIRD_PARTY_LICENSE.txt.
"""

import torch
import torch.nn as nn


class PhysicsAttentionIrregularMesh(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 32,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self, x: torch.Tensor, node_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, node_count, _ = x.shape

        fx_mid = self.in_project_fx(x).reshape(
            batch_size, node_count, self.heads, self.dim_head
        )
        fx_mid = fx_mid.permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(
            batch_size, node_count, self.heads, self.dim_head
        )
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )
        if node_mask is not None:
            slice_weights = slice_weights * node_mask[:, None, :, None].to(
                slice_weights.dtype
            )
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum(
            "bhnd,bhng->bhgd", fx_mid, slice_weights
        )
        slice_token = slice_token / (slice_norm[..., None] + 1.0e-5)

        query = self.to_q(slice_token)
        key = self.to_k(slice_token)
        value = self.to_v(slice_token)
        attention = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention = self.dropout(self.softmax(attention))
        out_slice_token = torch.matmul(attention, value)

        out = torch.einsum(
            "bhgd,bhng->bhnd", out_slice_token, slice_weights
        )
        out = out.permute(0, 2, 1, 3).reshape(batch_size, node_count, -1)
        out = self.to_out(out)
        if node_mask is not None:
            out = out * node_mask[..., None].to(out.dtype)
        return out
