"""Small building blocks shared by the backbone, encoder and decoder (ported from RT-DETR)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: str | nn.Module | None, inplace: bool = True) -> nn.Module:
    """Return an activation module by name (``None`` -> identity)."""
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    table = {"silu": nn.SiLU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU}
    if act.lower() not in table:
        raise ValueError(f"Unsupported activation: {act!r}")
    m = table[act.lower()]()
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed statistics and affine parameters (from DETR / torchvision)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps
        self.num_features = num_features

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        scale = (
            self.weight.reshape(1, -1, 1, 1)
            * (self.running_var.reshape(1, -1, 1, 1) + self.eps).rsqrt()
        )
        bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean.reshape(1, -1, 1, 1) * scale
        return x * scale + bias

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}"


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Bias value giving ``sigmoid(bias) == prior_prob``."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def deformable_attention_core_func(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pure-PyTorch multi-scale deformable attention.

    Args:
        value: ``[bs, value_length, n_head, c]``
        value_spatial_shapes: ``[(H_0, W_0), ...]`` per level.
        sampling_locations: ``[bs, query_length, n_head, n_levels, n_points, 2]`` in [0, 1].
        attention_weights: ``[bs, query_length, n_head, n_levels, n_points]``.

    Returns:
        ``[bs, query_length, n_head * c]``
    """
    bs, _, n_head, c = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        grid_l = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_value_list.append(
            F.grid_sample(
                value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        )
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, len_q, n_levels * n_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, len_q)
    )
    return output.permute(0, 2, 1)
