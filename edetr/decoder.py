"""Shared RT-DETR decoder with pluggable heads (by lyuwenyu, restructured).

Module layout and names mirror the official RT-DETR ``RTDETRTransformer`` (``input_proj``,
``enc_output``, ``enc_score_head``, ``enc_bbox_head``, ``query_pos_head``,
``dec_score_head``, ``dec_bbox_head``, ``denoising_class_embed``, ``decoder.layers``), so
the COCO checkpoint loads without key remapping. The only thing that differs between
RT-DETR and E-DETR is ``class_head`` (see :mod:`edetr.heads`); E-DETR-NIG
additionally sets ``box_uncertainty=True`` to add NIG heads.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .denoising import get_contrastive_denoising_training_group
from .heads import CLASS_HEADS, NIGHead
from .ops import MLP, deformable_attention_core_func, inverse_sigmoid


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention (box-relative sampling for 4-d reference points)."""

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile(
            [1, self.num_levels, self.num_points, 1]
        )
        grid_init *= torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        value = self.value_proj(value).reshape(bs, len_v, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = F.softmax(
            self.attention_weights(query).reshape(
                bs, len_q, self.num_heads, self.num_levels * self.num_points
            ),
            dim=-1,
        ).reshape(bs, len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            normalizer = (
                torch.tensor(value_spatial_shapes).flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            )
            sampling_locations = (
                reference_points.reshape(bs, len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        out = deformable_attention_core_func(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(out)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self, tgt, reference_points, memory, spatial_shapes, attn_mask=None, query_pos_embed=None
    ):
        q = k = tgt if query_pos_embed is None else tgt + query_pos_embed
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(
            tgt if query_pos_embed is None else tgt + query_pos_embed,
            reference_points,
            memory,
            spatial_shapes,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))


class TransformerDecoder(nn.Module):
    """Stack of decoder layers. Heads live on the parent (official RT-DETR layout)."""

    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])


class RTDETRDecoder(nn.Module):
    """RT-DETR decoder: query selection from encoder memory + iterative box refinement.

    Args:
        class_head: ``"logit"`` (RT-DETR) or ``"beta"`` (E-DETR).
        class_head_kwargs: Extra kwargs for the class head (e.g. Beta ``concentration``).
        box_uncertainty: Add NIG box-uncertainty heads (E-DETR-NIG).
        nig_head_kwargs: Extra kwargs for the NIG heads (e.g. ``init_sigma``).

    Outputs (dict): ``{class_key, "pred_boxes"[, "pred_nig"]}`` for the last layer, plus in
    training ``aux_outputs`` (earlier layers + encoder top-k) and ``dn_aux_outputs`` /
    ``dn_meta`` for contrastive denoising. ``class_key`` is the head's ``output_key``.
    Boxes are unactivated (apply ``sigmoid``) normalized cxcywh.
    """

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels=(256, 256, 256),
        feat_strides=(8, 16, 32),
        num_levels: int = 3,
        num_decoder_points: int = 4,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_idx: int = -1,
        eps: float = 1e-2,
        aux_loss: bool = True,
        class_head: str = "logit",
        class_head_kwargs: dict | None = None,
        box_uncertainty: bool = False,
        nig_head_kwargs: dict | None = None,
    ):
        super().__init__()
        feat_strides = list(feat_strides)
        assert len(feat_channels) <= num_levels and len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_decoder_layers + eval_idx
        self.aux_loss = aux_loss
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.box_uncertainty = box_uncertainty

        self._build_input_proj(list(feat_channels))

        layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points
        )
        self.decoder = TransformerDecoder(layer, num_decoder_layers)

        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )

        # encoder head (proposal generation)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        head_cls = CLASS_HEADS[class_head]
        head_kwargs = dict(class_head_kwargs or {})
        self.enc_score_head = head_cls(hidden_dim, num_classes, **head_kwargs)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder heads (one per layer)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.dec_score_head = nn.ModuleList(
            head_cls(hidden_dim, num_classes, **head_kwargs) for _ in range(num_decoder_layers)
        )
        self.dec_bbox_head = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)
        )
        if box_uncertainty:
            nig_kwargs = dict(nig_head_kwargs or {})
            self.enc_nig_head = NIGHead(hidden_dim, **nig_kwargs)
            self.dec_nig_head = nn.ModuleList(
                NIGHead(hidden_dim, **nig_kwargs) for _ in range(num_decoder_layers)
            )

        self.class_key = self.enc_score_head.output_key
        self._reset_parameters()

    def _reset_parameters(self):
        for mlp in [self.enc_bbox_head, *self.dec_bbox_head]:
            init.constant_(mlp.layers[-1].weight, 0)
            init.constant_(mlp.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj(self, feat_channels: list[int]) -> None:
        def proj(c_in, stride):
            conv = (
                nn.Conv2d(c_in, self.hidden_dim, 1, bias=False)
                if stride == 1
                else nn.Conv2d(c_in, self.hidden_dim, 3, 2, padding=1, bias=False)
            )
            return nn.Sequential(
                OrderedDict([("conv", conv), ("norm", nn.BatchNorm2d(self.hidden_dim))])
            )

        self.input_proj = nn.ModuleList(proj(c, 1) for c in feat_channels)
        c_in = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(proj(c_in, 2))
            c_in = self.hidden_dim

    # ---- encoder memory -> proposals ------------------------------------------------

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        n = len(proj_feats)
        for i in range(n, self.num_levels):
            proj_feats.append(self.input_proj[i](feats[-1] if i == n else proj_feats[-1]))
        flat, spatial_shapes = [], []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            flat.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(flat, 1), spatial_shapes

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device="cpu"):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing="ij"
            )
            grid_xy = (torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5) / torch.tensor(
                [w, h]
            ).to(dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, dn_class=None, dn_bbox_unact=None):
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)

        enc_class = self.enc_score_head(output_memory)
        enc_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(
            self.enc_score_head.rank_score(enc_class).max(-1).values, self.num_queries, dim=1
        )

        def gather(t):
            return t.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, t.shape[-1]))

        reference_points_unact = gather(enc_coord_unact)
        enc_topk_bboxes = reference_points_unact
        enc_topk_class = gather(enc_class)
        target = gather(output_memory)
        # Encoder NIG sees the (non-detached) top-k features, as in the reference E-DETR-NIG.
        enc_topk_nig = self.enc_nig_head(target) if self.box_uncertainty else None
        target = target.detach()

        if dn_bbox_unact is not None:
            reference_points_unact = torch.concat([dn_bbox_unact, reference_points_unact], 1)
        if dn_class is not None:
            target = torch.concat([dn_class, target], 1)
        return (
            target,
            reference_points_unact.detach(),
            enc_topk_bboxes,
            enc_topk_class,
            enc_topk_nig,
        )

    # ---- iterative refinement -----------------------------------------------------

    def _decode(self, target, ref_points_unact, memory, spatial_shapes, attn_mask=None):
        output = target
        out_bboxes, out_class, out_nig = [], [], []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_points = None

        for i, layer in enumerate(self.decoder.layers):
            query_pos_embed = self.query_pos_head(ref_points_detach)
            output = layer(
                output,
                ref_points_detach.unsqueeze(2),
                memory,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )

            inter_ref_bbox_unact = self.dec_bbox_head[i](output) + inverse_sigmoid(
                ref_points_detach
            )
            inter_ref_bbox = F.sigmoid(inter_ref_bbox_unact)

            if self.training:
                out_class.append(self.dec_score_head[i](output))
                if i == 0:
                    out_bboxes.append(inter_ref_bbox_unact)
                else:
                    # previous layer's non-detached box keeps the aux box-loss gradient path
                    out_bboxes.append(self.dec_bbox_head[i](output) + inverse_sigmoid(ref_points))
                if self.box_uncertainty:
                    out_nig.append(self.dec_nig_head[i](output))
            elif i == self.eval_idx:
                out_class.append(self.dec_score_head[i](output))
                out_bboxes.append(inter_ref_bbox_unact)
                if self.box_uncertainty:
                    out_nig.append(self.dec_nig_head[i](output))
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return (
            torch.stack(out_bboxes),
            torch.stack(out_class),
            torch.stack(out_nig) if out_nig else None,
        )

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        dn_class, dn_bbox_unact, attn_mask, dn_meta = None, None, None, None
        if self.training and self.num_denoising > 0:
            dn_class, dn_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )

        target, ref_unact, enc_topk_bboxes, enc_topk_class, enc_topk_nig = self._get_decoder_input(
            memory, spatial_shapes, dn_class, dn_bbox_unact
        )
        out_bboxes, out_class, out_nig = self._decode(
            target, ref_unact, memory, spatial_shapes, attn_mask
        )

        dn_bboxes = dn_class_out = dn_nig = None
        if self.training and dn_meta is not None:
            split = dn_meta["dn_num_split"]
            dn_bboxes, out_bboxes = torch.split(out_bboxes, split, dim=2)
            dn_class_out, out_class = torch.split(out_class, split, dim=2)
            if out_nig is not None:
                dn_nig, out_nig = torch.split(out_nig, split, dim=2)

        out = self._pack(out_class[-1], out_bboxes[-1], None if out_nig is None else out_nig[-1])
        if self.training and self.aux_loss:
            out["aux_outputs"] = [
                self._pack(out_class[i], out_bboxes[i], None if out_nig is None else out_nig[i])
                for i in range(len(out_class) - 1)
            ]
            out["aux_outputs"].append(self._pack(enc_topk_class, enc_topk_bboxes, enc_topk_nig))
            if dn_meta is not None:
                out["dn_aux_outputs"] = [
                    self._pack(dn_class_out[i], dn_bboxes[i], None if dn_nig is None else dn_nig[i])
                    for i in range(len(dn_class_out))
                ]
                out["dn_meta"] = dn_meta
        return out

    def _pack(self, cls_raw, boxes, nig_raw=None) -> dict:
        out = {self.class_key: self.enc_score_head.wrap(cls_raw), "pred_boxes": boxes}
        if nig_raw is not None:
            out["pred_nig"] = self.enc_nig_head.wrap(nig_raw)
        return out
