"""LVIS v1 detection data for VLM-EDETR: dataset, transforms, Repeat Factor Sampling, collate.

Labels are zero-indexed (``label = category_id - 1``); boxes are normalized cxcywh after the
transforms. Each sample also carries the image's verified-absent classes (``neg_labels``,
from ``neg_category_ids``) for the federated per-batch vocabulary. ``exclude_frequencies``
implements the OV-LVIS split: training drops annotations (and negatives) of the rare
classes, so rare-class AP (APr) measures zero-shot transfer.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import tv_tensors

from ..vlm.clip_backbone import CLIP_MEAN, CLIP_STD


def clip_transforms(train: bool, size: int = 640) -> T.Compose:
    """RT-DETR augmentation (zoom-out, IoU crop, flip) + square resize + CLIP normalization."""
    tail = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ]
    if not train:
        return T.Compose([T.Resize([size, size]), *tail, T.ConvertBoundingBoxFormat("CXCYWH")])
    return T.Compose(
        [
            T.RandomZoomOut(),
            T.RandomApply([T.RandomIoUCrop()], p=0.8),
            T.SanitizeBoundingBoxes(),
            T.RandomHorizontalFlip(),
            T.Resize([size, size]),
            *tail,
            T.SanitizeBoundingBoxes(),
            T.ConvertBoundingBoxFormat("CXCYWH"),
        ]
    )


class LvisDetection(Dataset):
    """LVIS v1 boxes. ``img_root`` holds ``train2017/`` and ``val2017/`` (COCO images)."""

    def __init__(
        self,
        img_root: str | Path,
        ann_path: str | Path,
        transforms=None,
        exclude_frequencies: Sequence[str] = (),
        limit: int | None = None,
    ):
        self.img_root = Path(img_root)
        self.ann_path = Path(ann_path)
        with open(ann_path) as f:
            data = json.load(f)
        self.imgs = {img["id"]: img for img in data["images"]}
        self.anns: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            self.anns[ann["image_id"]].append(ann)
        self.transforms = transforms
        self.excluded = {
            cat["id"] for cat in data["categories"] if cat.get("frequency") in exclude_frequencies
        }
        self.image_ids = sorted(self.imgs)[:limit]

    def __len__(self) -> int:
        return len(self.image_ids)

    def _annotations(self, image_id: int) -> list[dict]:
        return [a for a in self.anns[image_id] if a["category_id"] not in self.excluded]

    def per_image_category_ids(self) -> list[set[int]]:
        return [{a["category_id"] for a in self._annotations(i)} for i in self.image_ids]

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        info = self.imgs[image_id]
        # LVIS has no file_name: ".../val2017/000000397133.jpg" -> "val2017/000000397133.jpg"
        rel = "/".join(info["coco_url"].split("/")[-2:])
        img = Image.open(self.img_root / rel).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        for ann in self._annotations(image_id):
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann["category_id"] - 1)
        target = {
            "boxes": tv_tensors.BoundingBoxes(  # type: ignore[call-overload]  (Tensor subclass __new__)
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        rh, rw = target["boxes"].canvas_size  # (H, W) after the transforms
        neg = [c - 1 for c in info.get("neg_category_ids", []) if c not in self.excluded]
        return {
            "image": img,
            "image_id": image_id,
            "boxes": target["boxes"].as_subclass(torch.Tensor) / torch.tensor([rw, rh, rw, rh]),
            "labels": target["labels"],
            "neg_labels": torch.tensor(neg, dtype=torch.int64),
            "orig_size": torch.tensor([h, w]),
        }


def collate(samples: list[dict]) -> dict:
    """Stack images; keep per-image targets as lists (``targets["labels"][i]`` etc.)."""
    return {
        "images": torch.stack([s["image"] for s in samples]),
        "image_ids": [s["image_id"] for s in samples],
        "orig_sizes": torch.stack([s["orig_size"] for s in samples]),
        "targets": {k: [s[k] for s in samples] for k in ("boxes", "labels", "neg_labels")},
    }


class RepeatFactorSampler(Sampler[int]):
    """LVIS Repeat Factor Sampling (Gupta et al. 2019; detectron2), epoch-bounded.

    Category repeat factor ``r(c) = max(1, sqrt(t / f(c)))`` with ``f(c)`` the fraction of
    images containing ``c``; an image repeats ``max_c r(c)`` times per epoch (the fractional
    part stochastically). Call :meth:`set_epoch` each epoch to redraw and reshuffle.
    """

    def __init__(self, per_image_category_ids: list[set[int]], repeat_thresh: float = 0.001,
                 seed: int = 0):  # fmt: skip
        num_images = len(per_image_category_ids)
        freq: dict[int, int] = defaultdict(int)
        for cats in per_image_category_ids:
            for c in cats:
                freq[c] += 1
        rep = {c: max(1.0, math.sqrt(repeat_thresh / (n / num_images))) for c, n in freq.items()}
        factors = torch.tensor(
            [max((rep[c] for c in cats), default=1.0) for cats in per_image_category_ids],
            dtype=torch.float64,
        )
        self._int_part = torch.trunc(factors)
        self._frac_part = factors - self._int_part
        self._num_samples = int((self._int_part.sum() + self._frac_part.sum()).round().item())
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        reps = self._int_part + (torch.rand(len(self._frac_part), generator=g) < self._frac_part)
        indices = torch.repeat_interleave(torch.arange(len(reps)), reps.long())
        return iter(indices[torch.randperm(len(indices), generator=g)].tolist())

    def __len__(self) -> int:
        return self._num_samples


def build_lvis_loader(
    img_root: str,
    ann_path: str,
    train: bool,
    batch_size: int,
    num_workers: int = 8,
    exclude_frequencies: Sequence[str] = (),
    repeat_thresh: float = 0.001,
    seed: int = 0,
    limit: int | None = None,
) -> DataLoader:
    dataset = LvisDetection(
        img_root, ann_path, clip_transforms(train), exclude_frequencies, limit=limit
    )
    sampler = (
        RepeatFactorSampler(dataset.per_image_category_ids(), repeat_thresh, seed)
        if train
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=train,
        persistent_workers=num_workers > 0,
    )
