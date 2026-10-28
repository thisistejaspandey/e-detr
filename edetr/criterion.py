"""Hungarian matcher + set criterion for RT-DETR / E-DETR / E-DETR-NIG.

Loss terms (select via ``losses``; weight via ``weight_dict``):

- ``vfl`` -> ``loss_vfl``: varifocal BCE on ``pred_logits`` (RT-DETR).
- ``evidential_vfl`` -> ``loss_evidential_vfl``: cross-entropy of the Beta mean against the
  IoU target + variance term with the VFL weighting, plus an annealed KL evidence
  regulariser (E-DETR).
- ``boxes`` -> ``loss_bbox`` (L1) + ``loss_giou``.
- ``nig`` -> ``loss_nig``: NIG negative log-likelihood + evidence regulariser on the
  (detached) box mean (E-DETR-NIG).

Every term is applied to the final layer, each auxiliary layer (re-matched), the encoder
top-k proposals and the denoising groups (fixed CDN matching).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, box_iou, generalized_box_iou


def _xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")


def class_probs(outputs: dict) -> torch.Tensor:
    """Per-class probability ``[B, Q, C]`` from whichever class head produced ``outputs``."""
    if "pred_beta" in outputs:
        return outputs["pred_beta"].expectation
    return outputs["pred_logits"].sigmoid()


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class HungarianMatcher(nn.Module):
    """Focal-cost Hungarian matching on the final-layer predictions."""

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        assert cost_class or cost_bbox or cost_giou, "all costs can't be 0"
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        probs = class_probs(outputs)
        bs, num_queries = probs.shape[:2]
        out_prob = probs.flatten(0, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1).sigmoid()
        tgt_ids = torch.cat(list(targets["labels"]))
        tgt_bbox = torch.cat(list(targets["boxes"]))

        out_prob = out_prob[:, tgt_ids]
        neg = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos - neg
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(_xyxy(out_bbox), _xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).float().cpu()
        sizes = [len(v) for v in targets["boxes"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """DETR set loss with pluggable classification / box-uncertainty terms.

    Args:
        num_classes: Label-space size.
        matcher: :class:`HungarianMatcher`.
        weight_dict: ``{loss_name: weight}``; terms missing here are dropped.
        losses: Loss groups to compute (see module docstring).
        alpha / gamma: VFL negative weighting ``alpha * p^gamma``.
        kl_mode: Evidential regulariser form. ``"beta"`` (default) applies an independent
            per-class ``KL(Beta(m, 1) || Beta(1, 1)) = log m - (m - 1) / m``, consistent with the
            head's C independent binary Betas. ``"dirichlet"`` builds one C-way Dirichlet from
            each query's misleading-evidence parameters, so its scale depends on ``num_classes``.
        nig_reg_coeff: Weight of the NIG evidence regulariser ``|err| * (2v + alpha)``.

    ``forward(..., kl_weight=...)`` takes the (annealed) KL coefficient per step.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        losses: list[str],
        alpha: float = 0.75,
        gamma: float = 2.0,
        kl_mode: str = "beta",
        nig_reg_coeff: float = 1.0,
    ):
        super().__init__()
        if kl_mode not in ("dirichlet", "beta"):
            raise ValueError(f"kl_mode must be 'dirichlet' or 'beta', got {kl_mode!r}")
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.losses = list(losses)
        self.alpha = alpha
        self.gamma = gamma
        self.kl_mode = kl_mode
        self.nig_reg_coeff = nig_reg_coeff

    # ---- helpers --------------------------------------------------------------------

    @staticmethod
    def _src_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _vfl_targets(self, outputs, targets, indices, like: torch.Tensor):
        """One-hot ``target`` and IoU-valued ``target_score`` for the VFL family."""
        idx = self._src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx].sigmoid()
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets["boxes"], indices)], dim=0)
        ious = torch.diag(box_iou(_xyxy(src_boxes), _xyxy(target_boxes))).detach()

        target_classes = torch.full(
            like.shape[:2], self.num_classes, dtype=torch.int64, device=like.device
        )
        target_classes[idx] = torch.cat([t[i] for t, (_, i) in zip(targets["labels"], indices)])
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=like.dtype)
        target_score_o[idx] = ious.to(like.dtype)
        return target, target_score_o.unsqueeze(-1) * target

    # ---- loss terms -------------------------------------------------------------------

    def loss_vfl(self, outputs, targets, indices, num_boxes, **_):
        logits = outputs["pred_logits"]
        target, target_score = self._vfl_targets(outputs, targets, indices, logits)
        pred_score = logits.sigmoid().detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            logits, target_score, weight=weight, reduction="none"
        )
        return {"loss_vfl": loss.mean(1).sum() * logits.shape[1] / num_boxes}

    def loss_evidential_vfl(self, outputs, targets, indices, num_boxes, kl_weight=0.0, **_):
        beta = outputs["pred_beta"]
        p = beta.expectation
        target, target_score = self._vfl_targets(outputs, targets, indices, p)
        pred_score = p.detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        # cross-entropy of E against the IoU target (S-free) + variance term p(1-p)/(S+1)
        nll = -(target_score * torch.log(p + 1e-6) + (1 - target_score) * torch.log(1 - p + 1e-6))
        var = (pred_score * (1 - pred_score)) / (beta.concentration + 1)
        loss = (nll + var) * weight

        if kl_weight:
            # misleading evidence: beta-param for the target class, alpha-param otherwise
            m = torch.where(target.bool(), beta.alpha[..., 1], beta.alpha[..., 0])
            if self.kl_mode == "dirichlet":
                ones = torch.ones_like(m)
                sum_m = m.sum(-1, keepdim=True)
                kl = (
                    sum_m.lgamma()
                    - ones.sum(-1, keepdim=True).lgamma()
                    - (m.lgamma() - ones.lgamma()).sum(-1, keepdim=True)
                    + ((m - ones) * (m.digamma() - sum_m.digamma())).sum(-1, keepdim=True)
                )
            else:
                kl = torch.log(m) - (m - 1) / m
            loss = loss + torch.abs(target_score - pred_score) * kl * kl_weight

        return {"loss_evidential_vfl": loss.mean(1).sum() * p.shape[1] / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes, **_):
        idx = self._src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx].sigmoid()
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets["boxes"], indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = (
            1 - torch.diag(generalized_box_iou(_xyxy(src_boxes), _xyxy(target_boxes)))
        ).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_nig(self, outputs, targets, indices, num_boxes, **_):
        idx = self._src_idx(indices)
        # mean is detached: the NIG head learns uncertainty only; loss_boxes trains the mean
        src_boxes = outputs["pred_boxes"][idx].sigmoid().detach()
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets["boxes"], indices)], dim=0)
        nig = outputs["pred_nig"]
        v, a, b = nig.v[idx], nig.alpha[idx], nig.beta[idx]

        two_blambda = 2.0 * b * (1.0 + v)
        err = nig.box_errors(src_boxes, target_boxes)  # normalized or box-relative (nig.relative)
        err2 = err.pow(2)
        nll = (
            0.5 * torch.log(torch.pi / v)
            - a * torch.log(two_blambda)
            + (a + 0.5) * torch.log(v * err2 + two_blambda)
            + torch.lgamma(a)
            - torch.lgamma(a + 0.5)
        )
        reg = torch.abs(err) * (2.0 * v + a)
        return {
            "loss_nig": (nll + self.nig_reg_coeff * reg).sum() / num_boxes,
            "log_nig_nll": (nll.sum() / num_boxes).detach(),
            "log_nig_reg": (reg.sum() / num_boxes).detach() * self.nig_reg_coeff,
        }

    LOSS_FNS = {
        "vfl": loss_vfl,
        "evidential_vfl": loss_evidential_vfl,
        "boxes": loss_boxes,
        "nig": loss_nig,
    }

    # ---- forward ----------------------------------------------------------------------

    def _losses_for(self, outputs, targets, indices, num_boxes, suffix: str, **kw) -> dict:
        out = {}
        for name in self.losses:
            for k, v in self.LOSS_FNS[name](
                self, outputs, targets, indices, num_boxes, **kw
            ).items():
                if k in self.weight_dict:
                    out[k + suffix] = v * self.weight_dict[k]
                elif k.startswith("log_") and not suffix:
                    out[k] = v  # detached diagnostics, final layer only
        return out

    def forward(self, outputs, targets, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        main = {k: v for k, v in outputs.items() if "aux" not in k and k != "dn_meta"}
        indices = self.matcher(main, targets)

        num_boxes = torch.as_tensor(
            [sum(len(t) for t in targets["labels"])],
            dtype=torch.float,
            device=outputs["pred_boxes"].device,
        )
        if _world_size() > 1:
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / _world_size(), min=1).item()

        losses = self._losses_for(main, targets, indices, num_boxes, "", kl_weight=kl_weight)
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            losses.update(
                self._losses_for(
                    aux,
                    targets,
                    self.matcher(aux, targets),
                    num_boxes,
                    f"_aux_{i}",
                    kl_weight=kl_weight,
                )
            )
        if "dn_aux_outputs" in outputs:
            dn_meta = outputs["dn_meta"]
            dn_indices = self.cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = num_boxes * dn_meta["dn_num_group"]
            for i, aux in enumerate(outputs["dn_aux_outputs"]):
                losses.update(
                    self._losses_for(
                        aux, targets, dn_indices, dn_num_boxes, f"_dn_{i}", kl_weight=kl_weight
                    )
                )
        return losses

    @staticmethod
    def cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        device = targets["labels"][0].device
        out = []
        for i, num_gt in enumerate(len(t) for t in targets["labels"]):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                out.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                out.append((empty, empty))
        return out
