"""Pluggable classification / box-uncertainty heads for the shared RT-DETR decoder.

The decoder is identical for every model; only the heads differ:

- :class:`LogitHead` -- RT-DETR: per-class sigmoid logits, emitted as ``pred_logits``.
- :class:`BetaHead` -- E-DETR: per-class Beta(alpha, beta) evidence, emitted as a
  :class:`Beta` under ``pred_beta``.
- :class:`NIGHead` -- E-DETR-NIG: per-coordinate Normal-Inverse-Gamma box uncertainty,
  emitted as a :class:`NIG` under ``pred_nig`` (an extra head next to the class head).

A class head exposes ``output_key``, ``wrap(raw)`` (raw tensor -> what the losses consume)
and ``rank_score(raw)`` (per-class score used for encoder top-k query selection). Raw
tensors (not wrapped objects) flow through the decoder so the denoising split stays
model-agnostic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .ops import bias_init_with_prob

# --------------------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------------------


class Beta:
    """Per-class Beta distribution over object-ness.

    ``dist`` is ``[..., num_classes * 2]`` holding interleaved ``(alpha, beta)`` pairs
    (alpha = evidence for the class, beta = evidence against).

    - ``expectation = alpha / (alpha + beta)``
    - ``concentration = alpha + beta`` (total evidence ``S``)
    - ``uncertainty = 2 / S`` (clamped to [0, 1])
    - ``confidence = expectation * (1 - uncertainty)``
    """

    def __init__(self, dist: torch.Tensor, num_classes: int):
        self.dist = dist
        self.num_classes = num_classes

    @property
    def alpha(self) -> torch.Tensor:
        """``[..., C, 2]`` view of the (alpha, beta) parameters."""
        return self.dist.unflatten(-1, (self.num_classes, 2))

    @property
    def expectation(self) -> torch.Tensor:
        a, b = self.alpha.unbind(-1)
        return a / (a + b)

    @property
    def concentration(self) -> torch.Tensor:
        return self.alpha.sum(dim=-1)

    @property
    def uncertainty(self) -> torch.Tensor:
        return (2.0 / self.concentration.clamp_min(1.0)).clamp(0.0, 1.0)

    @property
    def confidence(self) -> torch.Tensor:
        return self.expectation * (1 - self.uncertainty)

    def score(self, score_type: str) -> torch.Tensor:
        if score_type == "expectation":
            return self.expectation
        if score_type == "confidence":
            return self.confidence
        raise ValueError(f"Unknown Beta score_type {score_type!r} (expectation|confidence)")

    @property
    def shape(self):
        return self.dist.shape

    def __getitem__(self, idx) -> Beta:
        return Beta(self.dist[idx], self.num_classes)

    def __len__(self) -> int:
        return len(self.dist)


class NIG:
    """Normal-Inverse-Gamma per box coordinate. ``dist`` is ``[..., 12]`` (4 coords x
    raw ``(v, alpha, beta)``); activations are applied lazily::

        v = softplus(v_raw) + 1e-6
        alpha = softplus(a_raw) + 1 + 1e-6
        beta = softplus(b_raw) + 1e-6

    ``relative=False`` (reference): the coordinates are the image-normalized cx, cy, w, h errors.
    ``relative=True``: they are box-relative deltas of the target from the predicted box,
    ``((gx - px) / pw, (gy - py) / ph, log(gw / pw), log(gh / ph))`` (Faster R-CNN encoding),
    so one sigma means the same thing for small and large boxes.
    """

    def __init__(self, dist: torch.Tensor, relative: bool = False):
        self.dist = dist
        self.relative = relative

    @property
    def raw_params(self) -> torch.Tensor:
        return self.dist.unflatten(-1, (4, 3))

    @property
    def v(self) -> torch.Tensor:
        return F.softplus(self.raw_params[..., 0]) + 1e-6

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_params[..., 1]) + 1.0 + 1e-6

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_params[..., 2]) + 1e-6

    @property
    def predictive_var(self) -> torch.Tensor:
        """Aleatoric variance ``E[sigma^2] = beta / (alpha - 1)`` scaled by ``1/v`` (Student-t)."""
        return self.beta / (self.v * (self.alpha - 1.0))

    @property
    def shape(self):
        return self.dist.shape

    def __getitem__(self, idx) -> NIG:
        return NIG(self.dist[idx], self.relative)

    def box_errors(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Errors in this NIG's coordinates; ``pred`` / ``target`` are normalized cxcywh."""
        if not self.relative:
            return target - pred
        wh = pred[..., 2:].clamp_min(1e-4)
        return torch.cat(
            [(target[..., :2] - pred[..., :2]) / wh, (target[..., 2:].clamp_min(1e-4) / wh).log()],
            -1,
        )

    def to_normalized_std(self, std: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Std in this NIG's coordinates -> std of normalized cx, cy, w, h (delta method)."""
        return std * pred[..., [2, 3, 2, 3]] if self.relative else std


# --------------------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------------------


class LogitHead(nn.Linear):
    """RT-DETR classification head: ``Linear(hidden, C)`` sigmoid logits.

    Subclasses ``nn.Linear`` directly so its state-dict keys (``weight``/``bias``) match
    the official RT-DETR checkpoint.
    """

    output_key = "pred_logits"

    def __init__(self, hidden_dim: int, num_classes: int, prior_prob: float = 0.01):
        super().__init__(hidden_dim, num_classes)
        self.num_classes = num_classes
        init.constant_(self.bias, bias_init_with_prob(prior_prob))

    def wrap(self, raw: torch.Tensor) -> torch.Tensor:
        return raw

    def rank_score(self, raw: torch.Tensor) -> torch.Tensor:
        return raw  # monotone in sigmoid(raw); matches the reference top-k on logits


class BetaHead(nn.Module):
    """E-DETR classification head: per-class Beta concentrations ``exp(Linear(x)) + 1``.

    Args:
        prior_prob / concentration: Bias init so every class starts at expectation
            ``prior_prob`` with total evidence ``concentration`` (focal-style prior).
        query_score: Beta statistic used to rank encoder proposals for top-k query
            selection (``expectation`` or ``confidence``).
    """

    output_key = "pred_beta"

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        prior_prob: float = 0.01,
        concentration: float = 200.0,
        query_score: str = "expectation",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.query_score = query_score
        self.dense = nn.Linear(hidden_dim, num_classes * 2)
        bias_alpha = math.log(max(prior_prob * concentration - 1.0, 1e-8))
        bias_beta = math.log(max((1.0 - prior_prob) * concentration - 1.0, 1e-8))
        with torch.no_grad():
            self.dense.bias[0::2] = bias_alpha
            self.dense.bias[1::2] = bias_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.dense(x)) + 1

    def wrap(self, raw: torch.Tensor) -> Beta:
        return Beta(raw, self.num_classes)

    def rank_score(self, raw: torch.Tensor) -> torch.Tensor:
        return Beta(raw, self.num_classes).score(self.query_score)

    @torch.no_grad()
    def init_from_logits(
        self, weight: torch.Tensor, bias: torch.Tensor, concentration: float = 200.0
    ):
        """Initialize from a trained sigmoid-logit head so ``expectation ~= sigmoid(z)``.

        With ``a = z/2 + c`` and ``b = -z/2 + c`` the Beta parameters are
        ``exp(a) + 1, exp(b) + 1``, whose expectation tends to ``sigmoid(a - b) = sigmoid(z)``
        for large ``c``. ``c = log(concentration / 2)`` puts total evidence ~``concentration``
        at ``z = 0`` (the reference prior's scale). E-DETR then starts from RT-DETR's exact
        ranking instead of a random head.
        """
        c = math.log(concentration / 2.0)
        self.dense.weight[0::2] = weight / 2
        self.dense.weight[1::2] = -weight / 2
        self.dense.bias[0::2] = bias / 2 + c
        self.dense.bias[1::2] = -bias / 2 + c


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


class NIGHead(nn.Module):
    """Box-uncertainty head: ``Linear(hidden, 12)`` raw NIG params, zero weight.

    With ``init_sigma=None`` (reference) the bias is zero too: v = 0.69, alpha = 1.69, beta = 0.69,
    a Student-t predictive std of ~1.6 image widths, about 100-1000x the matched box errors, so the
    head has to reach beta ~ e^-10 through its weights (the bias barely moves at lr 1e-4), which
    pulls the shared decoder features. ``init_sigma`` instead sets the bias so the predictive
    variance ``beta (1 + v) / (v (alpha - 1))`` starts at ``init_sigma**2`` (in the NIG's
    coordinates), with ``v = init_v`` and ``alpha = init_alpha``. ``relative`` selects box-relative
    coordinates (see :class:`NIG`).
    """

    output_key = "pred_nig"

    def __init__(
        self,
        hidden_dim: int,
        init_sigma: float | None = None,
        init_v: float = 1.0,
        init_alpha: float = 2.0,
        relative: bool = False,
    ):
        super().__init__()
        self.relative = relative
        self.linear = nn.Linear(hidden_dim, 12)
        init.constant_(self.linear.weight, 0)
        init.constant_(self.linear.bias, 0)
        if init_sigma is not None:
            beta = init_sigma**2 * init_v * (init_alpha - 1) / (1 + init_v)
            with torch.no_grad():
                self.linear.bias[0::3] = _inv_softplus(init_v)
                self.linear.bias[1::3] = _inv_softplus(init_alpha - 1)
                self.linear.bias[2::3] = _inv_softplus(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def wrap(self, raw: torch.Tensor) -> NIG:
        # fp32 like the Beta head's output: Accelerate only upcasts plain tensors, not NIG objects
        return NIG(raw.float(), self.relative)


CLASS_HEADS = {"logit": LogitHead, "beta": BetaHead}
