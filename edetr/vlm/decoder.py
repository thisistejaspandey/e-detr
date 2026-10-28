"""Open-vocabulary evidential decoder for VLM-EDETR (F-VLM / LaMI-DETR style).

The RT-DETR decoder with its class heads replaced: queries are projected into CLIP space by
one shared :class:`CLIPSpaceProjection`, scored against frozen CLIP text embeddings, and a
small per-layer :class:`ZeroShotBetaHead` maps each (query, class) similarity to Beta
``(alpha, beta)`` evidence. The class count is therefore set by the text embeddings at
runtime. Module names match the released checkpoints.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..decoder import TransformerDecoderLayer
from ..denoising import get_contrastive_denoising_training_group
from ..heads import Beta
from ..ops import MLP, inverse_sigmoid


def beta_bias_init(prior_prob: float, concentration: float) -> tuple[float, float]:
    """Biases giving ``exp(b) + 1`` Beta parameters with mean ``prior_prob`` and total
    evidence ``concentration``."""
    return (
        math.log(max(prior_prob * concentration - 1.0, 1e-8)),
        math.log(max((1.0 - prior_prob) * concentration - 1.0, 1e-8)),
    )


class CLIPSpaceProjection(nn.Module):
    """Project features to CLIP space and return temperature-scaled cosine similarities
    ``[..., C]`` against L2-normalized text embeddings ``[C, clip_dim]``. The learnable
    log-temperature starts at ``log(1 / 0.07)`` and is clamped to ``log(100)``."""

    def __init__(self, in_features: int, clip_dim: int, logit_scale_init: float = 2.659):
        super().__init__()
        self.proj = nn.Linear(in_features, clip_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        init.xavier_uniform_(self.proj.weight)
        init.constant_(self.proj.bias, 0.0)

    def forward(self, x, text_embeds):
        z = F.normalize(self.proj(x), dim=-1)
        return self.logit_scale.clamp(max=4.6055).exp() * (z @ text_embeds.t())


class ZeroShotBetaHead(nn.Module):
    """Shared scalar MLP ``1 -> hidden -> 2`` mapping each class similarity to Beta evidence.

    Applied per class, so it is independent of the vocabulary size. Output is
    ``[..., C * 2]`` interleaved ``(alpha, beta)`` (the :class:`Beta` layout). The last layer
    starts at zero weight, so every class begins at mean ``prior_prob`` with total evidence
    ``concentration``. Apply it to the selected queries only: it materializes
    ``[..., C, hidden]``.
    """

    def __init__(
        self,
        beta_hidden: int = 64,
        beta_layers: int = 3,
        prior_prob: float = 0.01,
        concentration: float = 200.0,
        max_logit: float = 20.0,
    ):
        super().__init__()
        self.beta = MLP(1, beta_hidden, 2, num_layers=beta_layers)
        self.max_logit = max_logit
        bias_alpha, bias_beta = beta_bias_init(prior_prob, concentration)
        with torch.no_grad():
            init.constant_(self.beta.layers[-1].weight, 0.0)
            self.beta.layers[-1].bias[0] = bias_alpha
            self.beta.layers[-1].bias[1] = bias_beta

    def forward(self, sim):
        logits = self.beta(sim.unsqueeze(-1)).clamp(max=self.max_logit)  # [..., C, 2]
        return (logits.exp() + 1.0).flatten(-2)


class FPNInputProjection(nn.Module):
    """1x1 conv + BN per encoder level, flattened into one memory sequence."""

    def __init__(self, feat_channels: list[int], hidden_dim: int, num_levels: int):
        super().__init__()
        projs = [self._proj(c, hidden_dim, 1) for c in feat_channels]
        c_in = feat_channels[-1]
        for _ in range(num_levels - len(feat_channels)):
            projs.append(self._proj(c_in, hidden_dim, 2))
            c_in = hidden_dim
        self.input_proj = nn.ModuleList(projs)

    @staticmethod
    def _proj(c_in: int, hidden_dim: int, stride: int) -> nn.Sequential:
        conv = (
            nn.Conv2d(c_in, hidden_dim, 1, bias=False)
            if stride == 1
            else nn.Conv2d(c_in, hidden_dim, 3, stride=stride, padding=1, bias=False)
        )
        return nn.Sequential(OrderedDict([("conv", conv), ("norm", nn.BatchNorm2d(hidden_dim))]))

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]
        for i in range(len(feats), len(self.input_proj)):
            proj.append(self.input_proj[i](proj[-1]))
        flat, spatial_shapes = [], []
        for feat in proj:
            _, _, h, w = feat.shape
            flat.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append((h, w))
        return torch.concat(flat, 1), spatial_shapes


def generate_anchors(spatial_shapes, eps=1e-2, grid_size=0.05, device="cpu"):
    """RT-DETR anchors (unactivated cxcywh) and their validity mask."""
    anchors = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(end=h, dtype=torch.float32),
            torch.arange(end=w, dtype=torch.float32),
            indexing="ij",
        )
        grid_xy = (torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5) / torch.tensor([w, h])
        wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
        anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
    anchors = torch.concat(anchors, 1).to(device)
    valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
    anchors = torch.log(anchors / (1 - anchors))
    return torch.where(valid_mask, anchors, torch.inf), valid_mask


class BetaEncoderHead(nn.Module):
    """Per memory location: projected content, CLIP similarities and anchor-relative box."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)

    def forward(self, memory, anchors, valid_mask, text_embeds, clip_proj):
        output_memory = self.enc_output(valid_mask.to(memory.dtype) * memory)
        enc_sims = clip_proj(output_memory, text_embeds)  # [B, L, C]
        enc_coord_unact = self.enc_bbox_head(output_memory) + anchors
        return output_memory, enc_sims, enc_coord_unact


class BetaProposalGenerator(nn.Module):
    """Top-K query selection by best text match, with LaMI-DETR language-conditioned queries.

    The top-K memory locations by best CLIP similarity become queries; only then is the Beta
    head applied (K queries, not all locations). With ``fuse_text_query`` each query's content
    is its (detached) visual feature plus a projection of its best-matched text embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        eps: float,
        clip_dim: int,
        beta_hidden: int = 64,
        fuse_text_query: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.eps = eps
        self.beta_enc_head = BetaEncoderHead(hidden_dim)
        self.enc_sim_to_beta = ZeroShotBetaHead(beta_hidden=beta_hidden)
        self.text_query_proj = (
            nn.Sequential(nn.Linear(clip_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            if fuse_text_query
            else None
        )

    def forward(self, memory, spatial_shapes, text_embeds, clip_proj, dn_class=None, dn_bbox=None):
        anchors, valid_mask = generate_anchors(spatial_shapes, self.eps, device=memory.device)
        output_memory, enc_sims, enc_coord_unact = self.beta_enc_head(
            memory, anchors, valid_mask, text_embeds, clip_proj
        )
        _, topk_ind = torch.topk(enc_sims.max(-1).values, self.num_queries, dim=1)

        def gather(t):
            return t.gather(1, topk_ind.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

        sel_memory = gather(output_memory)  # [B, K, hidden]
        sel_sims = gather(enc_sims)  # [B, K, C]
        reference_points_unact = gather(enc_coord_unact)  # [B, K, 4]
        enc_topk_bboxes = reference_points_unact
        enc_topk_beta = self.enc_sim_to_beta(sel_sims)  # [B, K, C * 2]

        target = sel_memory.detach()
        if self.text_query_proj is not None:
            target = target + self.text_query_proj(text_embeds[sel_sims.argmax(-1)])

        if dn_class is not None:
            target = torch.concat([dn_class, target], 1)
            reference_points_unact = torch.concat([dn_bbox, reference_points_unact], 1)
        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_beta, sel_memory


class DenoisingHandler(nn.Module):
    """Holds the denoising class embedding (indexed by absolute class id)."""

    def __init__(self, num_denoising, num_classes, num_queries, hidden_dim, label_noise_ratio,
                 box_noise_scale):  # fmt: skip
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

    def prepare(self, targets):
        return get_contrastive_denoising_training_group(
            targets,
            self.num_classes,
            self.num_queries,
            self.class_embed,
            num_denoising=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
        )


class TransformerBetaDecoder(nn.Module):
    """Decoder layers with per-layer box heads and CLIP-space Beta score heads."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, beta_hidden=64):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.score_head = nn.ModuleList(
            ZeroShotBetaHead(beta_hidden=beta_hidden) for _ in range(num_layers)
        )
        self.bbox_head = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_layers)
        )
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for mlp in self.bbox_head:
            init.constant_(mlp.layers[-1].weight, 0)
            init.constant_(mlp.layers[-1].bias, 0)

    def forward(self, tgt, ref_points_unact, memory, spatial_shapes, text_embeds, clip_proj,
                attn_mask=None):  # fmt: skip
        """Returns stacked ``(boxes, beta_dists, hidden_states)`` over the decoded layers
        (all layers in training; only ``eval_idx`` at inference)."""
        output = tgt
        out_bboxes, out_logits, hidden = [], [], []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_points = None

        for i, layer in enumerate(self.layers):
            query_pos_embed = self.query_pos_head(ref_points_detach)
            output = layer(
                output,
                ref_points_detach.unsqueeze(2),
                memory,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )
            inter_ref_bbox_unact = self.bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            inter_ref_bbox = F.sigmoid(inter_ref_bbox_unact)

            if self.training:
                out_logits.append(self.score_head[i](clip_proj(output, text_embeds)))
                hidden.append(output)
                if i == 0:
                    out_bboxes.append(inter_ref_bbox_unact)
                else:
                    # previous layer's non-detached box keeps the aux box-loss gradient path
                    out_bboxes.append(self.bbox_head[i](output) + inverse_sigmoid(ref_points))
            elif i == self.eval_idx:
                out_logits.append(self.score_head[i](clip_proj(output, text_embeds)))
                out_bboxes.append(inter_ref_bbox_unact)
                hidden.append(output)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(out_bboxes), torch.stack(out_logits), torch.stack(hidden)


class BetaTransformer(nn.Module):
    """Proposal generation + iterative refinement, emitting ``pred_beta`` / ``pred_boxes``
    (unactivated cxcywh) and, in training, ``aux_outputs`` (earlier layers + encoder top-k),
    ``dn_aux_outputs`` and ``dn_meta``. ``_hidden_states`` / ``_enc_hidden`` are returned for
    the NIG heads."""

    def __init__(
        self,
        num_classes=1203,
        hidden_dim=256,
        clip_dim=1024,
        num_queries=300,
        feat_channels=(256, 256, 256),
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        beta_hidden=64,
        logit_scale_init=2.659,
        fuse_text_query=True,
    ):
        super().__init__()
        self.aux_loss = aux_loss
        self.num_denoising = num_denoising
        self.clip_proj = CLIPSpaceProjection(hidden_dim, clip_dim, logit_scale_init)
        self.fpn = FPNInputProjection(list(feat_channels), hidden_dim, num_levels)
        self.proposal_generator = BetaProposalGenerator(
            hidden_dim, num_queries, eps, clip_dim, beta_hidden, fuse_text_query
        )
        self.denoising = DenoisingHandler(
            num_denoising, num_classes, num_queries, hidden_dim, label_noise_ratio,
            box_noise_scale,
        )  # fmt: skip
        layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points
        )
        self.decoder = TransformerBetaDecoder(
            hidden_dim, layer, num_decoder_layers, eval_idx, beta_hidden
        )

    def forward(self, feats, text_embeds, targets=None):
        dn_class = dn_bbox = attn_mask = dn_meta = None
        if self.training and self.num_denoising > 0:
            dn_class, dn_bbox, attn_mask, dn_meta = self.denoising.prepare(targets)

        memory, spatial_shapes = self.fpn(feats)
        target, ref_unact, enc_topk_bboxes, enc_topk_beta, enc_hidden = self.proposal_generator(
            memory, spatial_shapes, text_embeds, self.clip_proj, dn_class, dn_bbox
        )
        out_bboxes, out_logits, hidden = self.decoder(
            target, ref_unact, memory, spatial_shapes, text_embeds, self.clip_proj, attn_mask
        )

        dn_bboxes = dn_logits = None
        if dn_meta is not None:
            dn_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

        def beta(dist):
            return Beta(dist, dist.shape[-1] // 2)

        out = {"pred_beta": beta(out_logits[-1]), "pred_boxes": out_bboxes[-1]}
        if self.training and self.aux_loss:
            out["aux_outputs"] = [
                {"pred_beta": beta(c), "pred_boxes": b}
                for c, b in zip(out_logits[:-1], out_bboxes[:-1])
            ]
            out["aux_outputs"].append(
                {"pred_beta": beta(enc_topk_beta), "pred_boxes": enc_topk_bboxes}
            )
            if dn_meta is not None:
                out["dn_aux_outputs"] = [
                    {"pred_beta": beta(c), "pred_boxes": b} for c, b in zip(dn_logits, dn_bboxes)
                ]
                out["dn_meta"] = dn_meta
        out["_hidden_states"] = hidden
        out["_enc_hidden"] = enc_hidden
        return out
