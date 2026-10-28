"""PResNet (ResNet-vd) backbone from RT-DETR (by lyuwenyu). Parameter names match the
official COCO checkpoint, so it loads without remapping."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import ConvNormLayer, FrozenBatchNorm2d, get_activation

RESNET_CFG = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}

IMAGENET_URL = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="b"):
        super().__init__()
        stride1, stride2 = (stride, 1) if variant == "a" else (1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(ch_out, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2c(self.branch2b(self.branch2a(x)))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="b"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=i != 0,
                    variant=variant,
                    act=act,
                )
            )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PResNet(nn.Module):
    """ResNet-vd backbone returning multi-scale features.

    Args:
        depth: 18 / 34 / 50 / 101.
        return_idx: Stage indices to return (``[1, 2, 3]`` -> strides 8/16/32).
        freeze_at: Freeze the stem plus this many stages (``-1`` freezes nothing).
        freeze_norm: Replace every BatchNorm with :class:`FrozenBatchNorm2d`.
        pretrained: Download ImageNet weights. Unnecessary when initializing from a
            COCO detector checkpoint, which overwrites them anyway.
    """

    def __init__(
        self,
        depth: int = 50,
        variant: str = "d",
        num_stages: int = 4,
        return_idx: tuple[int, ...] = (1, 2, 3),
        act: str = "relu",
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()
        block_nums = RESNET_CFG[depth]
        ch_in = 64
        if variant in ("c", "d"):
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
        self.conv1 = nn.Sequential(
            OrderedDict(
                (name, ConvNormLayer(c_in, c_out, k, s, act=act))
                for c_in, c_out, k, s, name in conv_def
            )
        )

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        out_channels = [block.expansion * v for v in ch_out_list]
        out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], i + 2, act=act, variant=variant)
            )
            ch_in = out_channels[i]

        self.return_idx = list(return_idx)
        self.out_channels = [out_channels[i] for i in self.return_idx]
        self.out_strides = [out_strides[i] for i in self.return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])
        if freeze_norm:
            self._freeze_norm(self)
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(IMAGENET_URL[depth]))

    @staticmethod
    def _freeze_parameters(m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module) -> nn.Module:
        if isinstance(m, nn.BatchNorm2d):
            return FrozenBatchNorm2d(m.num_features)
        for name, child in m.named_children():
            new_child = self._freeze_norm(child)
            if new_child is not child:
                setattr(m, name, new_child)
        return m

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
