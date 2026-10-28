"""Minimal training / inference loop pieces for VLM-EDETR (used by ``tools/``)."""

from __future__ import annotations

import logging
import math
import re
from copy import deepcopy
from typing import cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """FP32 shadow copy updated as ``ema = d * ema + (1 - d) * model`` with
    ``d = decay * (1 - exp(-updates / warmups))`` (ramps up over the first steps)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmups = warmups
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]


NO_DECAY = re.compile(r"^(?:encoder|decoder)\..*(?:bias|norm.*weight)$")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW over trainable parameters; no weight decay on encoder/decoder biases and norms."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if NO_DECAY.match(name) else decay).append(p)
    groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, weight_decay=weight_decay)


def kl_weight_at(step: int, steps_per_epoch: int, end_value: float, end_epoch: int) -> float:
    """KL coefficient annealed linearly from 0 at step 0 to ``end_value`` at ``end_epoch``."""
    if end_epoch <= 0:
        return end_value
    return end_value * min(1.0, step / (end_epoch * steps_per_epoch))


def train_one_epoch(model, criterion, loader, optimizer, scheduler, accelerator, ema, epoch,
                    kl_end: float, kl_end_epoch: int, max_norm: float = 0.1,
                    log_every: int = 50) -> None:  # fmt: skip
    model.train()
    steps_per_epoch = len(loader)
    for i, batch in enumerate(loader):
        step = epoch * steps_per_epoch + i
        kl_weight = kl_weight_at(step, steps_per_epoch, kl_end, kl_end_epoch)
        outputs = model(batch["images"], batch["targets"])
        loss_dict = criterion(outputs, batch["targets"], kl_weight=kl_weight)
        loss = cast(torch.Tensor, sum(v for v in loss_dict.values() if v.requires_grad))
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss is {loss.item()}: {loss_dict}")

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        if max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(accelerator.unwrap_model(model))

        if i % log_every == 0 and accelerator.is_main_process:
            logger.info(
                f"epoch {epoch} [{i}/{steps_per_epoch}] loss {loss.item():.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e} kl {kl_weight:.2e}"
            )


@torch.no_grad()
def predict(model, postprocessor, loader, accelerator, score_type: str | None = None) -> list[dict]:
    """LVIS-format detections (absolute xywh, ``category_id = label + 1``) for every image of
    ``loader``, gathered over processes. Per-detection extras (Beta expectation and
    concentration, NIG ``box_std``) are included."""
    model.eval()
    per_image = []
    for batch in loader:
        outputs = model(batch["images"])
        dets = postprocessor(outputs, batch["image_ids"], batch["orig_sizes"], score_type)
        per_image += [{"image_id": d.image_id, "dets": d.to_coco(with_extras=True)} for d in dets]
    per_image = accelerator.gather_for_metrics(per_image, use_gather_object=True)
    return [det for entry in per_image for det in entry["dets"]]


def lvis_ap(ann_path: str, predictions: list[dict], image_ids: list[int] | None = None) -> dict:
    """LVIS federated box AP (AP, APr/c/f, ...; 300 detections per image) with hotcoco, a
    fast drop-in for the lvis-api evaluator. ``image_ids`` restricts scoring to the
    predicted images."""
    import hotcoco

    gt = hotcoco.LVIS(str(ann_path))  # type: ignore[attr-defined]
    evaluator = hotcoco.LVISeval(gt, gt.loadRes(predictions), "bbox")  # type: ignore[attr-defined]
    if image_ids is not None:
        # `params` is returned by copy: `evaluator.params.imgIds = ...` would be silently lost
        params = evaluator.params
        params.imgIds = list(image_ids)
        evaluator.params = params
    evaluator.run()
    return {k: float(v) for k, v in evaluator.get_results().items()}
