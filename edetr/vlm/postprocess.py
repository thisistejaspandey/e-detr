"""Top-k detections for VLM-EDETR with the F-VLM detector / frozen-CLIP score ensemble."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..postprocess import Detections


class VLMPostProcessor:
    """``score = det ** (1 - a) * clip ** a`` per class, then top-k over queries x classes.

    ``det`` is the Beta ``expectation`` or ``confidence``; ``clip`` is the frozen-CLIP region
    score (``outputs["vlm_scores"]``). As in F-VLM the exponent is per class: ``base_alpha``
    for classes seen in training, ``novel_alpha`` for ``novel_class_ids`` (zero-indexed; the
    LVIS rare classes in the OV-LVIS split). Without ``novel_class_ids`` the single
    ``roi_alpha`` applies to every class. Without ``vlm_scores`` the detector score is used.

    ``score_type="uncertainty_weighted"`` is the uncertainty-guided fusion: the exponent is
    the Beta's own per-(query, class) uncertainty ``u = 2 / S`` instead of a fixed ``a``,
    ``score = confidence ** (1 - u) * clip ** u``, so the less evidence the detector has, the
    more the score defers to frozen CLIP.
    """

    def __init__(
        self,
        num_classes: int = 1203,
        num_top_queries: int = 300,
        score_type: str = "confidence",
        roi_alpha: float = 0.65,
        base_alpha: float = 0.20,
        novel_alpha: float = 0.15,
        novel_class_ids: Sequence[int] | None = None,
    ):
        self.num_top_queries = num_top_queries
        self.score_type = score_type
        self.roi_alpha = roi_alpha
        self.alpha_vec = None
        if novel_class_ids is not None:
            novel = torch.zeros(num_classes, dtype=torch.bool)
            novel[torch.as_tensor(list(novel_class_ids), dtype=torch.long)] = True
            self.alpha_vec = torch.where(
                novel, torch.tensor(float(novel_alpha)), torch.tensor(float(base_alpha))
            )

    @torch.no_grad()
    def __call__(
        self,
        outputs: dict,
        image_ids: Sequence[int],
        image_sizes: torch.Tensor,
        score_type: str | None = None,
    ) -> list[Detections]:
        """``image_sizes`` is ``[B, 2]`` original (H, W)."""
        beta = outputs["pred_beta"]
        score_type = score_type or self.score_type
        vlm_scores = outputs.get("vlm_scores")
        if score_type == "uncertainty_weighted":
            scores_full = beta.confidence
            if vlm_scores is not None:
                u = beta.uncertainty
                scores_full = scores_full.clamp_min(0).pow(1 - u) * vlm_scores.clamp_min(0).pow(u)
        elif vlm_scores is not None:
            scores_full = beta.score(score_type)
            a = self.roi_alpha if self.alpha_vec is None else self.alpha_vec.to(scores_full.device)
            scores_full = scores_full.clamp_min(0).pow(1 - a) * vlm_scores.clamp_min(0).pow(a)
        else:
            scores_full = beta.score(score_type)
        extras_full = {"expectation": beta.expectation, "concentration": beta.concentration}

        num_classes = scores_full.shape[-1]
        scores, index = torch.topk(scores_full.flatten(1), self.num_top_queries, dim=-1)
        labels = index % num_classes
        query = index // num_classes
        boxes = outputs["pred_boxes"].sigmoid().gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
        extras = {k: v.flatten(1).gather(1, index) for k, v in extras_full.items()}
        if "pred_nig" in outputs:
            q4 = query.unsqueeze(-1).expand(-1, -1, 4)
            extras["box_std"] = outputs["pred_nig"].predictive_var.gather(1, q4).sqrt()

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
