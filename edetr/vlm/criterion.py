"""Set criterion for VLM-EDETR / VLM-NIG: evidential VFL over the per-batch vocabulary,
L1 + GIoU boxes, and the NIG box-uncertainty loss.

With a per-batch vocabulary the model scores queries against a subset of the text
embeddings, so target labels are remapped into the subset's index space and the
classification loss is restricted to each image's supervised classes (``vocab_mask``:
its positives + selected verified-absent negatives). LVIS is federated, so a class an image
was not checked for must not be trained as absent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from ..criterion import HungarianMatcher, SetCriterion, _xyxy


class VLMSetCriterion(SetCriterion):
    """Loss groups: ``evidential_vfl``, ``boxes``, ``nig``. ``forward(..., kl_weight=...)``
    takes the annealed KL coefficient. ``kl_mode`` is as in :class:`~edetr.criterion.SetCriterion`:
    ``"beta"`` (default) is the per-class Beta KL; ``"dirichlet"`` is the C'-way form the
    released VLM-EDETR checkpoint was trained with."""

    def __init__(self, matcher: HungarianMatcher, weight_dict, losses, alpha=0.75, gamma=2.0,
                 kl_mode="beta", nig_reg_coeff=1.0):  # fmt: skip
        super().__init__(
            num_classes=0,  # set per batch by the vocabulary
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            kl_mode=kl_mode,
            nig_reg_coeff=nig_reg_coeff,
        )

    def loss_evidential_vfl(self, outputs, targets, indices, num_boxes, kl_weight=0.0,
                            vocab_mask=None, **_):  # fmt: skip
        beta = outputs["pred_beta"]
        num_classes = beta.num_classes
        idx = self._src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx].sigmoid()
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets["boxes"], indices)], dim=0)
        ious = torch.diag(box_iou(_xyxy(src_boxes), _xyxy(target_boxes))).detach()

        p = beta.expectation
        target_classes = torch.full(p.shape[:2], num_classes, dtype=torch.int64, device=p.device)
        target_classes[idx] = torch.cat([t[i] for t, (_, i) in zip(targets["labels"], indices)])
        target = F.one_hot(target_classes, num_classes=num_classes + 1)[..., :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=p.dtype)
        target_score_o[idx] = ious.to(p.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = p.detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
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

        if vocab_mask is not None:
            loss = loss * vocab_mask.unsqueeze(1)
        return {"loss_evidential_vfl": loss.mean(1).sum() * p.shape[1] / num_boxes}

    def loss_nig(self, outputs, targets, indices, num_boxes, **kw):
        if "pred_nig" not in outputs:  # denoising queries carry no NIG
            return {}
        return super().loss_nig(outputs, targets, indices, num_boxes, **kw)

    LOSS_FNS = {
        "evidential_vfl": loss_evidential_vfl,
        "boxes": SetCriterion.loss_boxes,
        "nig": loss_nig,
    }

    def forward(self, outputs, targets, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        vocab = outputs.get("vocab")
        vocab_mask = None
        if vocab is not None:
            pos_of = torch.full((int(vocab.max()) + 1,), -1, dtype=torch.int64, device=vocab.device)
            pos_of[vocab] = torch.arange(vocab.numel(), device=vocab.device)
            targets = dict(targets)
            targets["labels"] = [pos_of[lbl.to(vocab.device)] for lbl in targets["labels"]]
            vocab_mask = outputs["vocab_mask"]

        main = {k: v for k, v in outputs.items() if "aux" not in k and k != "dn_meta"}
        indices = self.matcher(main, targets)
        num_boxes = torch.as_tensor(
            [sum(len(t) for t in targets["labels"])],
            dtype=torch.float,
            device=outputs["pred_boxes"].device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = num_boxes / torch.distributed.get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        kw = {"kl_weight": kl_weight, "vocab_mask": vocab_mask}
        losses = self._losses_for(main, targets, indices, num_boxes, "", **kw)
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux, targets)
            losses.update(self._losses_for(aux, targets, aux_indices, num_boxes, f"_aux_{i}", **kw))
        if "dn_aux_outputs" in outputs:
            dn_meta = outputs["dn_meta"]
            dn_indices = self.cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = num_boxes * dn_meta["dn_num_group"]
            for i, aux in enumerate(outputs["dn_aux_outputs"]):
                losses.update(
                    self._losses_for(aux, targets, dn_indices, dn_num_boxes, f"_dn_{i}", **kw)
                )
        return losses
