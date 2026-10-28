"""Build VLM-EDETR components from a config (see ``configs/vlm_edetr_lvis.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ..criterion import HungarianMatcher
from .criterion import VLMSetCriterion
from .model import VLMEdetr, VLMNIGEdetr
from .postprocess import VLMPostProcessor


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """YAML config + ``key=value`` overrides. ``defaults_from: other.yaml`` (relative to the
    file) loads that config first and merges this one on top."""
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    if "defaults_from" in cfg:
        base = load_config(str(Path(path).parent / cfg.pop("defaults_from")))
        cfg = OmegaConf.merge(base, cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides or []))
    assert isinstance(cfg, DictConfig)
    return cfg


def build_model(cfg: DictConfig) -> VLMEdetr:
    m = cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True))
    cls = VLMNIGEdetr if m.pop("nig") else VLMEdetr
    return cls(**m)


def build_criterion(cfg: DictConfig) -> VLMSetCriterion:
    c = cfg.criterion
    return VLMSetCriterion(
        matcher=HungarianMatcher(**c.matcher),
        weight_dict=dict(c.weight_dict),
        losses=list(c.losses),
        alpha=c.alpha,
        gamma=c.gamma,
        kl_mode=c.kl_mode,
        nig_reg_coeff=c.nig_reg_coeff,
    )


def build_postprocessor(cfg: DictConfig, num_classes: int) -> VLMPostProcessor:
    """Rare (novel) LVIS classes are read from the val annotation's frequency tags."""
    import json

    with open(cfg.data.val_ann) as f:
        cats = json.load(f)["categories"]
    novel = [c["id"] - 1 for c in cats if c.get("frequency") == "r"]
    p = cfg.postprocess
    return VLMPostProcessor(
        num_classes=num_classes,
        num_top_queries=p.num_top_queries,
        score_type=p.score_type,
        base_alpha=p.base_alpha,
        novel_alpha=p.novel_alpha,
        novel_class_ids=novel,
    )
