"""VLM-EDETR: open-vocabulary E-DETR (F-VLM / LaMI-DETR style) and its NIG variant.

A frozen CLIP ResNet supplies the feature pyramid; a trainable RT-DETR HybridEncoder and the
evidential open-vocabulary decoder (:mod:`edetr.vlm.decoder`) localize and score queries
against frozen CLIP text embeddings. At inference the frozen-CLIP region scores of the
predicted boxes are returned as ``vlm_scores`` for the F-VLM ensemble in
:class:`~edetr.vlm.postprocess.VLMPostProcessor`.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detector import TrainableMixin
from ..encoder import HybridEncoder
from ..heads import NIG, NIGHead
from ..weights import read_state_dict
from .clip_backbone import CLIPBackbone
from .decoder import BetaTransformer

MULTI_SCALE = [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]


class VLMEdetr(TrainableMixin):
    """VLM-EDETR.

    Args:
        clip_model_name: CLIP ResNet variant (RN50 in the paper).
        text_embeddings_path: ``.pt`` with the ``[C, clip_dim]`` class text embeddings (see
            ``tools/build_text_embeddings.py``); sets the vocabulary size. Alternatively pass
            ``num_classes`` and load the embeddings with a checkpoint.
        vlm_temperature: Softmax temperature of the frozen-CLIP region scores.
        batch_vocabulary: Train over a per-batch vocabulary subset (positives + per-image
            verified-absent negatives) instead of all classes (LaMI-DETR). Inference always
            uses the full vocabulary.
        num_neg: Verified-absent negatives kept per image, the ``num_neg`` least similar (by
            CLIP text cosine) to the image's positives.
        fuse_text_query: LaMI-DETR language-conditioned query initialization.
        multi_scale: Training-time square resize sizes (``None`` disables).
        train_only: If set, freeze everything except these module paths.
    """

    text_embeddings: torch.Tensor

    def __init__(
        self,
        clip_model_name: str = "RN50",
        text_embeddings_path: str | None = None,
        num_classes: int | None = None,
        logit_scale_init: float = 2.659,
        vlm_temperature: float = 0.01,
        batch_vocabulary: bool = True,
        num_neg: int = 50,
        fuse_text_query: bool = True,
        multi_scale: Sequence[int] | None = tuple(MULTI_SCALE),
        train_only: Sequence[str] | None = None,
    ):
        super().__init__()
        self.vlm_temperature = vlm_temperature
        self.batch_vocabulary = batch_vocabulary
        self.num_neg = num_neg
        self.multi_scale = list(multi_scale) if multi_scale else None

        self.clip = CLIPBackbone(clip_model_name)
        clip_dim = self.clip.embed_dim
        if text_embeddings_path is not None:
            text_embeddings = read_text_embeddings(text_embeddings_path, clip_dim)
        elif num_classes is not None:
            text_embeddings = torch.zeros(num_classes, clip_dim)
        else:
            raise ValueError("Provide either text_embeddings_path or num_classes.")
        self.num_classes = text_embeddings.shape[0]
        self.register_buffer("text_embeddings", text_embeddings)

        self.encoder = HybridEncoder(
            in_channels=(512, 1024, 2048),
            feat_strides=(8, 16, 32),
            hidden_dim=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            enc_act="gelu",
            use_encoder_idx=(-1,),
            num_encoder_layers=1,
            expansion=1.0,
            depth_mult=1.0,
            act="silu",
        )
        self.decoder = BetaTransformer(
            num_classes=self.num_classes,
            hidden_dim=256,
            clip_dim=clip_dim,
            num_queries=300,
            num_decoder_layers=6,
            num_denoising=100,
            logit_scale_init=logit_scale_init,
            fuse_text_query=fuse_text_query,
        )
        self._build_extra_heads()
        self.set_trainable(train_only=train_only)

    def _build_extra_heads(self) -> None:
        """Hook for subclasses to add heads before freezing is applied."""

    def train(self, mode: bool = True):
        """Fully frozen submodules (always CLIP; with ``train_only`` also the encoder and
        decoder) stay in eval mode. A frozen decoder then runs as at inference (last layer
        only, no denoising queries), so a head trained on top of it, like the VLM-NIG head,
        sees exactly the inference-time queries."""
        super().train(mode)
        for module in self.children():
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self

    def forward(self, images, targets=None):
        if self.training and self.multi_scale:
            size = int(random.choice(self.multi_scale))
            images = F.interpolate(images, size=[size, size])

        c3, c4, c5 = self.clip.forward_features(images)
        feats = self.encoder([c3, c4, c5])

        vocab = vocab_mask = None
        if self.training and self.batch_vocabulary and targets is not None:
            vocab, vocab_mask = self._build_batch_vocabulary(targets)
        text = self.text_embeddings if vocab is None else self.text_embeddings[vocab]
        out = self.decoder(feats, text, targets)
        if vocab is not None:
            out["vocab"] = vocab
            out["vocab_mask"] = vocab_mask

        self._add_extra_outputs(out, c5)
        out.pop("_hidden_states")
        out.pop("_enc_hidden")
        if not self.training:
            out["vlm_scores"] = self._vlm_region_scores(c5, out["pred_boxes"])
        return out

    def _add_extra_outputs(self, out: dict, c5: torch.Tensor) -> None:
        """Hook for subclasses; ``out`` still holds the decoder hidden states."""

    def _build_batch_vocabulary(self, targets):
        """Per-batch vocabulary ``V`` (sorted absolute class ids ``[C']``) and per-image
        supervision mask ``[B, C']``: ``mask[i, v] = 1`` iff ``V[v]`` is a positive of image i or
        one of its selected verified-absent negatives. ``(None, None)`` for a label-free batch.
        """
        te = self.text_embeddings
        device = te.device
        labels = [lbl.to(device) for lbl in targets["labels"]]
        negs = targets.get("neg_labels")

        selected_neg: list[torch.Tensor] = []
        for i, pos in enumerate(labels):
            neg = negs[i].to(device) if negs is not None and negs[i].numel() else pos.new_zeros(0)
            if neg.numel() == 0 or self.num_neg <= 0:
                selected_neg.append(neg[: max(self.num_neg, 0)])
            elif pos.numel() == 0 or neg.numel() <= self.num_neg:
                selected_neg.append(neg[: self.num_neg])
            else:
                # keep the negatives least similar to any positive (fp32: bf16 ties change ranks)
                with torch.autocast(device_type=device.type, enabled=False):
                    sim = (te[neg].float() @ te[pos].float().t()).max(dim=1).values
                selected_neg.append(neg[torch.topk(sim, self.num_neg, largest=False).indices])

        vocab = torch.unique(torch.cat(labels + selected_neg).long())
        if vocab.numel() == 0:
            return None, None
        pos_of = torch.full((self.num_classes,), -1, dtype=torch.long, device=device)
        pos_of[vocab] = torch.arange(vocab.numel(), device=device)
        mask = te.new_zeros(len(labels), vocab.numel())
        for i, (pos, neg) in enumerate(zip(labels, selected_neg)):
            mask[i, pos_of[pos]] = 1.0
            mask[i, pos_of[neg]] = 1.0
        return vocab, mask

    @torch.no_grad()
    def _vlm_region_scores(self, layer4_map, pred_boxes):
        """Frozen-CLIP region class probabilities ``[B, K, C]``: softmax at ``vlm_temperature``
        over the text embeddings plus an implicit zero background logit (dropped)."""
        roi = self.clip.encode_rois(layer4_map, pred_boxes.sigmoid())
        sims = roi @ self.text_embeddings.t()
        bg = sims.new_zeros(*sims.shape[:-1], 1)
        return (torch.cat([sims, bg], dim=-1) / self.vlm_temperature).softmax(-1)[..., :-1]


class VLMNIGEdetr(VLMEdetr):
    """VLM-EDETR + a NIG box-uncertainty head (``pred_nig``) on the decoder hidden states.

    One :class:`~edetr.heads.NIGHead` is shared by all decoder layers and the encoder top-k
    queries; it is not applied to denoising queries. The paper recipe trains only this head
    (``train_only=["nig_embed"]``) on top of a trained VLM-EDETR.
    """

    def _build_extra_heads(self) -> None:
        head = NIGHead(256)
        self.nig_embed = nn.ModuleList([head for _ in range(self.decoder.decoder.num_layers)])

    def _add_extra_outputs(self, out: dict, c5: torch.Tensor) -> None:
        pad = out["dn_meta"]["dn_num_split"][0] if "dn_meta" in out else 0
        hidden = out["_hidden_states"]  # [layers, B, dn + Q, hidden]
        nig = torch.stack([h(hs) for h, hs in zip(self.nig_embed, hidden)])[:, :, pad:]
        out["pred_nig"] = NIG(nig[-1])
        if self.training and "aux_outputs" in out:
            for i, aux in enumerate(out["aux_outputs"][:-1]):
                aux["pred_nig"] = NIG(nig[i])
            out["aux_outputs"][-1]["pred_nig"] = NIG(self.nig_embed[-1](out["_enc_hidden"]))


def read_text_embeddings(path: str, clip_dim: int) -> torch.Tensor:
    emb = torch.load(path, map_location="cpu")
    emb = emb["text_embeddings"] if isinstance(emb, dict) else emb
    emb = F.normalize(emb.float(), dim=-1)
    if emb.ndim != 2 or emb.shape[1] != clip_dim:
        raise ValueError(f"text embeddings {tuple(emb.shape)} don't match CLIP dim {clip_dim}")
    return emb


def load_vlm_checkpoint(model: VLMEdetr, path: str) -> None:
    """Load VLM-EDETR / VLM-NIG weights (raw state dict, ``{"model": ...}`` or an EMA
    ``{"ema": {"module": ...}}`` checkpoint). Strict, except that NIG heads may be absent (a
    VLM-NIG model initialized from VLM-EDETR) and the unused ``zeroshot_embed`` head of the
    original VLM-NIG checkpoints is ignored."""
    state = {k: v for k, v in read_state_dict(path).items() if not k.startswith("zeroshot_embed.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("nig_embed.")]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )


# Prompt ensemble (CLIP ImageNet templates) used for the class text embeddings.
PROMPT_TEMPLATES = [
    "{}.", "a photo of a {}.", "a bad photo of a {}.", "a photo of many {}.",
    "a sculpture of a {}.", "a photo of the hard to see {}.",
    "a low resolution photo of the {}.", "a rendering of a {}.",
    "graffiti of a {}.", "a bad photo of the {}.", "a cropped photo of the {}.",
    "a tattoo of a {}.", "the embroidered {}.", "a photo of a hard to see {}.",
    "a bright photo of a {}.", "a photo of a clean {}.", "a photo of a dirty {}.",
    "a dark photo of the {}.", "a drawing of a {}.", "a photo of my {}.",
    "the plastic {}.", "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.", "a sculpture of the {}.",
    "a bright photo of the {}.", "a cropped photo of a {}.", "a plastic {}.",
    "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.",
    "a rendering of the {}.", "a {} in a video game.", "a photo of one {}.",
    "a doodle of a {}.", "a close-up photo of the {}.", "the origami {}.",
    "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.",
    "a origami {}.", "a low resolution photo of a {}.", "the toy {}.",
    "a rendition of the {}.", "a photo of the clean {}.", "a photo of a large {}.",
    "a rendition of a {}.", "a photo of a nice {}.", "a photo of a weird {}.",
    "a blurry photo of a {}.", "a cartoon {}.", "art of a {}.",
    "a sketch of the {}.", "a embroidered {}.", "a pixelated photo of a {}.",
    "itap of the {}.", "a jpeg corrupted photo of the {}.", "a good photo of a {}.",
    "a plushie {}.", "a photo of the nice {}.", "a photo of the small {}.",
    "a photo of the weird {}.", "the cartoon {}.", "art of the {}.",
    "a drawing of the {}.", "a photo of the large {}.",
    "a black and white photo of a {}.", "the plushie {}.", "a dark photo of a {}.",
    "itap of a {}.", "graffiti of the {}.", "a toy {}.", "itap of my {}.",
    "a photo of a cool {}.", "a photo of a small {}.", "a tattoo of the {}.",
]  # fmt: skip


def clean_class_name(name: str) -> str:
    """``aerosol_can`` -> ``aerosol can``; ``ice_(food)`` -> ``ice food`` (LVIS names)."""
    name = name.replace("_", " ").replace("(", "").replace(")", "")
    return re.sub(r"\s+", " ", name).strip()


@torch.no_grad()
def build_text_embeddings(
    class_names: list[str],
    clip_model_name: str = "RN50",
    templates: list[str] | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> torch.Tensor:
    """CLIP text embeddings ``[C, D]`` over a prompt ensemble: the raw per-template embeddings
    are averaged, then L2-normalized once (CLIP / F-VLM recipe)."""
    import clip

    templates = templates or PROMPT_TEMPLATES
    names = [clean_class_name(n) for n in class_names]
    model, _ = clip.load(clip_model_name, device=device, jit=False)
    model.eval()
    per_template = []
    for tmpl in templates:
        prompts = [tmpl.format(n) for n in names]
        raw = [
            model.encode_text(clip.tokenize(prompts[i : i + batch_size]).to(device)).float().cpu()
            for i in range(0, len(prompts), batch_size)
        ]
        per_template.append(torch.cat(raw))
    return F.normalize(torch.stack(per_template).mean(0), dim=-1)
