"""Frozen OpenAI CLIP ResNet image encoder used as the VLM-EDETR backbone.

The same frozen pass provides the detection feature pyramid (``layer2..layer4``) and, at
inference, the F-VLM region embeddings (RoI-Align on ``layer4`` + the frozen ``attnpool``).
"""

from __future__ import annotations

from typing import Any

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, roi_align

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPBackbone(nn.Module):
    """Frozen CLIP ResNet (RN50, RN50x4, ...). Weights are kept in fp32 so mixed precision
    is governed by autocast, not by the stored dtype.

    Args:
        model_name: CLIP architecture (ResNet variants only).
        device: Device ``clip.load`` builds on (the module still moves with ``.to``).
    """

    def __init__(self, model_name: str = "RN50", device: str | None = None) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load(model_name, device=device, jit=False)
        self.visual = model.visual.float()
        self.model_name = model_name
        self.embed_dim: int = model.visual.output_dim
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(c3, c4, c5) = (layer2, layer3, layer4)`` of CLIP-normalized images; for RN50
        ``[512, 1024, 2048]`` channels at strides ``[8, 16, 32]``."""
        v: Any = self.visual
        x = x.to(next(v.parameters()).dtype)
        x = v.relu1(v.bn1(v.conv1(x)))
        x = v.relu2(v.bn2(v.conv2(x)))
        x = v.relu3(v.bn3(v.conv3(x)))
        x = v.avgpool(x)
        x = v.layer1(x)
        c3 = v.layer2(x)
        c4 = v.layer3(c3)
        c5 = v.layer4(c4)
        return c3.float(), c4.float(), c5.float()

    @torch.no_grad()
    def encode_rois(self, layer4_map: torch.Tensor, boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        """F-VLM region embeddings: RoI-Align normalized cxcywh boxes ``[B, K, 4]`` on the
        ``layer4`` map to the attnpool grid (7x7 for RN50), then apply the frozen ``attnpool``.

        Returns ``[B, K, D]`` L2-normalized embeddings in CLIP text space.
        """
        v: Any = self.visual
        B, K, _ = boxes_cxcywh.shape
        _, _, fH, fW = layer4_map.shape
        res = int(round((v.attnpool.positional_embedding.shape[0] - 1) ** 0.5))

        boxes_xyxy = box_convert(boxes_cxcywh, "cxcywh", "xyxy").clamp(0.0, 1.0)
        scale = boxes_xyxy.new_tensor([fW, fH, fW, fH])
        roi = roi_align(
            layer4_map.float(),
            [boxes_xyxy[b] * scale for b in range(B)],
            output_size=res,
            spatial_scale=1.0,
            aligned=True,
        )  # [B*K, C, res, res]
        feats = v.attnpool(roi.to(v.conv1.weight.dtype)).float()
        return F.normalize(feats, dim=-1).view(B, K, -1)
