"""Turn decoder outputs into per-image detections (+ per-detection uncertainty)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torchvision.ops import box_convert

from .heads import Beta


@dataclass
class Detections:
    """Top-k detections for one image.

    ``boxes`` are normalized cxcywh; ``labels`` are zero-indexed. ``extras`` holds
    per-detection uncertainty arrays aligned with ``scores``:

    - ``expectation`` / ``concentration`` (E-DETR): Beta mean and total evidence ``S``.
    - ``box_std`` (E-DETR-NIG): ``[K, 4]`` predictive std of normalized cx, cy, w, h.
    """

    image_id: int
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    image_size: torch.Tensor  # original (H, W)
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_coco(
        self, category_ids: Sequence[int] | None = None, with_extras: bool = False
    ) -> list[dict]:
        """COCO result dicts (absolute xywh). ``category_ids[label]`` gives the dataset's
        category id (``None``: ``label + 1``, contiguous 1-indexed ids as in LVIS)."""
        h, w = (float(v) for v in self.image_size)
        xywh = box_convert(self.boxes.float(), "cxcywh", "xywh") * torch.tensor(
            [w, h, w, h], device=self.boxes.device
        )
        labels = self.labels.cpu().tolist()
        cats = (
            [category_ids[lbl] for lbl in labels] if category_ids else [lbl + 1 for lbl in labels]
        )
        out = []
        extras = (
            {k: v.float().cpu().tolist() for k, v in self.extras.items()} if with_extras else {}
        )
        for i, (box, cat, score) in enumerate(
            zip(xywh.cpu().tolist(), cats, self.scores.float().cpu().tolist())
        ):
            det = {
                "image_id": int(self.image_id),
                "bbox": box,
                "category_id": int(cat),
                "score": score,
            }
            for k, v in extras.items():
                det[k] = v[i]
            out.append(det)
        return out


class PostProcessor:
    """Select the top-k (query, class) pairs over all queries x classes.

    Args:
        num_top_queries: Detections kept per image.
        score_type: Beta statistic used as the detection score for E-DETR
            (``expectation`` or ``confidence``); ignored for RT-DETR (sigmoid).
    """

    def __init__(self, num_top_queries: int = 300, score_type: str = "expectation"):
        self.num_top_queries = num_top_queries
        self.score_type = score_type

    @torch.no_grad()
    def __call__(
        self,
        outputs: dict,
        image_ids: Sequence[int],
        image_sizes: torch.Tensor,
        score_type: str | None = None,
    ) -> list[Detections]:
        """``image_sizes`` is ``[B, 2]`` original (H, W), used to denormalize boxes."""
        score_type = score_type or self.score_type
        boxes = outputs["pred_boxes"].sigmoid()
        extras_full: dict[str, torch.Tensor] = {}
        if "pred_beta" in outputs:
            beta: Beta = outputs["pred_beta"]
            scores_full = beta.score(score_type)
            extras_full["expectation"] = beta.expectation
            extras_full["concentration"] = beta.concentration
        else:
            scores_full = outputs["pred_logits"].sigmoid()
        num_classes = scores_full.shape[-1]

        scores, index = torch.topk(scores_full.flatten(1), self.num_top_queries, dim=-1)
        labels = index % num_classes
        query = index // num_classes
        boxes = boxes.gather(1, query.unsqueeze(-1).expand(-1, -1, 4))

        extras = {k: v.flatten(1).gather(1, index) for k, v in extras_full.items()}
        if "pred_nig" in outputs:
            nig = outputs["pred_nig"]
            q4 = query.unsqueeze(-1).expand(-1, -1, 4)
            std = nig.predictive_var.gather(1, q4).sqrt()  # [B, K, 4] in the NIG's coordinates
            extras["box_std"] = nig.to_normalized_std(std, boxes)

        return [
            Detections(
                image_id=int(image_ids[b]),
                boxes=boxes[b],
                labels=labels[b],
                scores=scores[b],
                image_size=image_sizes[b],
                extras={k: v[b] for k, v in extras.items()},
            )
            for b in range(len(image_ids))
        ]
