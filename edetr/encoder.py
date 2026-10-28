"""RT-DETR HybridEncoder (AIFI transformer on the top level + CCFM fusion), by lyuwenyu.
Parameter names match the official COCO checkpoint."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import ConvNormLayer, get_activation


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class CSPRepLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, act="silu"
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(hidden, hidden, act=act) for _ in range(num_blocks)]
        )
        self.conv3 = (
            ConvNormLayer(hidden, out_channels, 1, 1, bias=bias, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, src, pos_embed=None):
        q = k = src if pos_embed is None else src + pos_embed
        attn, _ = self.self_attn(q, k, value=src)
        src = self.norm1(src + self.dropout1(attn))
        ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(ffn))


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, pos_embed=None):
        for layer in self.layers:
            src = layer(src, pos_embed=pos_embed)
        return src


def sincos_position_embedding(w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0):
    grid_w, grid_h = torch.meshgrid(
        torch.arange(int(w), dtype=torch.float32),
        torch.arange(int(h), dtype=torch.float32),
        indexing="ij",
    )
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos embeddings"
    pos_dim = embed_dim // 4
    omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]
    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None]


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature

        self.input_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim)
            )
            for c in self.in_channels
        )
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, enc_act
        )
        self.encoder = nn.ModuleList(
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in self.use_encoder_idx
        )

        n = len(self.in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(n - 1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion
                )
            )
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(n - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion
                )
            )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                pos_embed = sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(src.device)
                memory = self.encoder[i](src, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                )

        n = len(self.in_channels)
        inner_outs = [proj_feats[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = self.lateral_convs[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            upsample = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            inner_outs.insert(
                0,
                self.fpn_blocks[n - 1 - idx](torch.concat([upsample, proj_feats[idx - 1]], dim=1)),
            )

        outs = [inner_outs[0]]
        for idx in range(n - 1):
            downsample = self.downsample_convs[idx](outs[-1])
            outs.append(
                self.pan_blocks[idx](torch.concat([downsample, inner_outs[idx + 1]], dim=1))
            )
        return outs
