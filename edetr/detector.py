"""RT-DETR-family detector: PResNet backbone -> HybridEncoder -> shared decoder.

One class covers RT-DETR, E-DETR and E-DETR-NIG; they differ only in decoder heads::

    RT-DETR     class_head="logit"
    E-DETR      class_head="beta"
    E-DETR-NIG  class_head="beta", box_uncertainty=True
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

import torch.nn as nn
import torch.nn.functional as F

from .backbone import PResNet
from .decoder import RTDETRDecoder
from .encoder import HybridEncoder

logger = logging.getLogger(__name__)


class TrainableMixin(nn.Module):
    """``set_trainable(freeze, train_only)`` + a ``train()`` that keeps frozen BN in eval."""

    _frozen_bn: list[nn.Module]

    def _resolve(self, path: str) -> nn.Module:
        module = self.get_submodule(path)
        if not isinstance(module, nn.Module):
            raise TypeError(f"{path!r} is not a module")
        return module

    def set_trainable(
        self, freeze: Sequence[str] = (), train_only: Sequence[str] | None = None
    ) -> None:
        """Freeze ``freeze`` paths (or everything but ``train_only``) and pin frozen BN stats.

        ``requires_grad=False`` doesn't stop BatchNorm running stats from updating in
        ``train()`` mode, so every BatchNorm whose parameters are all frozen is kept in
        ``eval()`` (see :meth:`train`).
        """
        for path in freeze:
            for p in self._resolve(path).parameters():
                p.requires_grad = False
        if train_only is not None:
            for p in self.parameters():
                p.requires_grad = False
            for path in train_only:
                for p in self._resolve(path).parameters():
                    p.requires_grad = True

        self._frozen_bn = [
            m
            for m in self.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
            and not any(p.requires_grad for p in m.parameters())
        ]
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Trainable params: {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M "
            f"(freeze={list(freeze)}, train_only={train_only}, "
            f"frozen BN layers={len(self._frozen_bn)})"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        for m in getattr(self, "_frozen_bn", []):
            m.eval()
        return self


# RT-DETR's COCO multi-scale training sizes (square). 640 is repeated to weight it 3x.
RTDETR_COCO_SCALES = [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]


class Detector(TrainableMixin):
    """RT-DETR / E-DETR / E-DETR-NIG.

    Args:
        num_classes: Size of the label space.
        class_head: ``"logit"`` or ``"beta"``.
        class_head_kwargs: Passed to the class head (e.g. ``{"query_score": "confidence"}``).
        box_uncertainty: Add NIG box-uncertainty heads.
        nig_head_kwargs: Passed to the NIG heads (e.g. ``{"init_sigma": 0.01}``).
        multi_scale: Training-time batch resize sizes (ints = square, ``[h, w]`` pairs
            allowed). ``None`` disables multi-scale (the batch keeps its loader size).
        freeze: Module paths to freeze (e.g. ``["backbone", "encoder"]``).
        train_only: If set, freeze everything except these module paths
            (e.g. ``["decoder.enc_nig_head", "decoder.dec_nig_head"]``). Applied after ``freeze``.
        depth: Backbone depth (18/34/50/101).
        num_queries / num_denoising: Decoder query counts.
    """

    def __init__(
        self,
        num_classes: int,
        class_head: str = "logit",
        class_head_kwargs: dict | None = None,
        box_uncertainty: bool = False,
        nig_head_kwargs: dict | None = None,
        multi_scale: Sequence[int | Sequence[int]] | None = None,
        freeze: Sequence[str] = (),
        train_only: Sequence[str] | None = None,
        depth: int = 50,
        num_queries: int = 300,
        num_denoising: int = 100,
        backbone_freeze_at: int = 0,
        imagenet_pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = PResNet(
            depth=depth,
            variant="d",
            return_idx=(1, 2, 3),
            freeze_at=backbone_freeze_at,
            freeze_norm=True,
            pretrained=imagenet_pretrained_backbone,
        )
        self.encoder = HybridEncoder(
            in_channels=self.backbone.out_channels,
            feat_strides=(8, 16, 32),
            hidden_dim=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            enc_act="gelu",
            use_encoder_idx=(2,),
            num_encoder_layers=1,
            expansion=1.0 if depth >= 50 else 0.5,
            depth_mult=1.0,
            act="silu",
        )
        self.decoder = RTDETRDecoder(
            num_classes=num_classes,
            num_queries=num_queries,
            feat_channels=(256, 256, 256),
            feat_strides=(8, 16, 32),
            num_levels=3,
            num_decoder_layers=6,
            num_denoising=num_denoising,
            class_head=class_head,
            class_head_kwargs=class_head_kwargs,
            box_uncertainty=box_uncertainty,
            nig_head_kwargs=nig_head_kwargs,
        )
        self.multi_scale = list(multi_scale) if multi_scale else None
        self._frozen_bn: list[nn.Module] = []
        self.set_trainable(freeze=freeze, train_only=train_only)

    @property
    def class_key(self) -> str:
        return self.decoder.class_key

    def forward(self, images, targets=None, mask=None):
        """``mask`` (padding, from padded batches) is unused: RT-DETR trains on fixed squares."""
        if self.training and self.multi_scale:
            size = random.choice(self.multi_scale)
            size = [int(s) for s in size] if isinstance(size, Sequence) else [int(size)] * 2
            images = F.interpolate(images, size=size)
        return self.decoder(self.encoder(self.backbone(images)), targets)
