"""Draw detections with their evidential uncertainty (needs matplotlib).

Each box is labelled ``name score u=… a=… b=…``: the detection score, the Beta uncertainty
``u = 2 / S`` and the Beta parameters ``a = α = E·S`` and ``b = β = (1 − E)·S`` of the
predicted class. Colors are per class.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .postprocess import Detections

PALETTE = [
    "#f94144", "#277da1", "#f9c74f", "#43aa8b", "#f3722c",
    "#9b5de5", "#90be6d", "#f15bb5", "#4d908e", "#00bbf9",
]  # fmt: skip


def draw_detections(
    image,
    det: Detections,
    class_names: Sequence[str],
    score_threshold: float = 0.3,
    ax=None,
):
    """Draw ``det`` over ``image`` (PIL image or ``[H, W, 3]`` array at the original size)
    and return the matplotlib axes."""
    import matplotlib.pyplot as plt
    from matplotlib import patches

    if ax is None:
        _, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    h, w = (float(v) for v in det.image_size)
    cx, cy, bw, bh = (det.boxes.float().cpu() * torch.tensor([w, h, w, h])).unbind(-1)
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    extras = {k: v.float().cpu() for k, v in det.extras.items()}

    keep = (det.scores.cpu() >= score_threshold).nonzero().flatten().tolist()
    for i in sorted(keep, key=lambda i: float(det.scores[i])):  # most confident on top
        x1, y1, x2, y2 = boxes[i].tolist()
        label_id = int(det.labels[i])
        color = PALETTE[label_id % len(PALETTE)]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"
            )
        )
        label = f"{class_names[label_id]} {float(det.scores[i]):.2f}"
        if "expectation" in extras:
            e, conc = float(extras["expectation"][i]), float(extras["concentration"][i])
            label += (
                f" u={min(1.0, 2 / max(conc, 1.0)):.2f} a={e * conc:.2f} b={(1 - e) * conc:.2f}"
            )
        ax.text(
            x1,
            y1 - 4,
            label,
            color="black",
            fontsize=9,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1, "edgecolor": "none"},
        )
    return ax
