"""Initialize a :class:`Detector` from the official RT-DETR COCO checkpoint
(``rtdetr_r50vd_6x_coco_from_paddle.pth`` from https://github.com/lyuwenyu/RT-DETR).

- Backbone / encoder / decoder trunk load 1:1 (module names match the checkpoint).
- Class heads (encoder top-k head, 6 decoder heads, denoising class embedding):
  * logit head, same classes  -> copied.
  * beta head                 -> converted from the trained logits via
                                 :meth:`BetaHead.init_from_logits` (E-DETR starts at
                                 RT-DETR's ranking instead of a random head).
  * different label space     -> rows picked from COCO by ``class_map`` (target label ->
                                 COCO zero-indexed label, ``None`` = keep random init);
                                 without a map the heads stay randomly initialized.
- NIG heads are never in the checkpoint and keep their zero init.

Any other missing/unexpected key raises, so silent partial loads can't happen.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from .detector import Detector
from .heads import BetaHead

logger = logging.getLogger(__name__)

CLASS_HEAD_PREFIXES = ("decoder.enc_score_head", "decoder.dec_score_head")
DN_EMBED = "decoder.denoising_class_embed.weight"
NIG_PREFIXES = ("decoder.enc_nig_head", "decoder.dec_nig_head")


def read_state_dict(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("ema"), dict) and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("module"), dict):  # ModelEMA.state_dict()
        state = ckpt["module"]
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        state = ckpt["model"]
    else:
        state = ckpt
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = v
    return out


def _class_head_names(model: Detector) -> list[str]:
    return ["decoder.enc_score_head"] + [
        f"decoder.dec_score_head.{i}" for i in range(len(model.decoder.dec_score_head))
    ]


def load_rtdetr_coco(
    model: Detector,
    path: str,
    class_map: Sequence[int | None] | None = None,
    beta_concentration: float = 200.0,
) -> dict[str, int]:
    """Load the RT-DETR COCO checkpoint into ``model`` (see module docstring).

    Returns a summary ``{"trunk": n, "class_heads": n, "random_init": n}``.
    """
    src = read_state_dict(path)
    tgt = model.state_dict()
    num_classes = model.num_classes
    src_classes = src["decoder.enc_score_head.weight"].shape[0]

    if num_classes != src_classes and class_map is not None and len(class_map) != num_classes:
        raise ValueError(f"class_map has {len(class_map)} entries for {num_classes} classes")
    if num_classes == src_classes and class_map is None:
        class_map = list(range(num_classes))

    # 1) trunk: everything except class heads / dn embedding
    def is_head(k: str) -> bool:
        return k.startswith(CLASS_HEAD_PREFIXES) or k == DN_EMBED

    trunk = {k: v for k, v in src.items() if not is_head(k)}
    unexpected = [k for k in trunk if k not in tgt]
    shape_mismatch = [k for k in trunk if k in tgt and tgt[k].shape != trunk[k].shape]
    if unexpected or shape_mismatch:
        raise RuntimeError(
            f"Checkpoint trunk mismatch: unexpected={unexpected[:5]} shape={shape_mismatch[:5]}"
        )
    missing = model.load_state_dict(trunk, strict=False).missing_keys
    leftover = [k for k in missing if not (is_head(k) or k.startswith(NIG_PREFIXES))]
    if leftover:
        raise RuntimeError(f"Model keys not covered by the checkpoint: {leftover[:10]}")

    # 2) class heads + denoising embedding, row-selected through class_map
    # class_map None (and num_classes != 80): the heads keep their own init
    n_heads, n_random = 0, num_classes
    if class_map is not None:
        rows = [(t, s) for t, s in enumerate(class_map) if s is not None]
        n_random = num_classes - len(rows)
        t_idx = torch.tensor([t for t, _ in rows], dtype=torch.long)
        s_idx = torch.tensor([s for _, s in rows], dtype=torch.long)
        with torch.no_grad():
            for name in _class_head_names(model):
                head = model.get_submodule(name)
                w_src, b_src = src[f"{name}.weight"], src[f"{name}.bias"]
                if isinstance(head, BetaHead):
                    w = torch.zeros(num_classes, w_src.shape[1])
                    b = torch.full((num_classes,), -4.595)  # logit(0.01) for unmapped rows
                    w[t_idx], b[t_idx] = w_src[s_idx], b_src[s_idx]
                    head.init_from_logits(w, b, concentration=beta_concentration)
                else:
                    head.weight[t_idx] = w_src[s_idx].to(head.weight.dtype)
                    head.bias[t_idx] = b_src[s_idx].to(head.bias.dtype)
                n_heads += 1
            emb = model.decoder.denoising_class_embed.weight
            emb[t_idx] = src[DN_EMBED][s_idx]
            emb[num_classes] = src[DN_EMBED][src_classes]  # padding row
    logger.info(
        f"Loaded RT-DETR COCO weights from {path}: trunk={len(trunk)} tensors, "
        f"class heads={n_heads} ({type(model.decoder.enc_score_head).__name__}), "
        f"classes from COCO={num_classes - n_random}/{num_classes}"
    )
    return {"trunk": len(trunk), "class_heads": n_heads, "random_init": n_random}
