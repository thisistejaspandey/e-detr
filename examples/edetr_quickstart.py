"""E-DETR quickstart: build from RT-DETR COCO weights, take one training step, run inference.

    python examples/edetr_quickstart.py --weights rtdetr_r50vd_6x_coco_from_paddle.pth

The RT-DETR R50 COCO checkpoint is released at https://github.com/lyuwenyu/RT-DETR. E-DETR's
Beta heads are initialized from its trained logit heads (expectation = sigmoid(logit)), so the
model starts at RT-DETR's accuracy. Plug ``model`` / ``criterion`` / ``postprocessor`` into
your own detection training loop; targets are normalized cxcywh boxes and zero-indexed labels.
"""

import argparse

import torch

from edetr.criterion import HungarianMatcher, SetCriterion
from edetr.detector import RTDETR_COCO_SCALES, Detector
from edetr.postprocess import PostProcessor
from edetr.weights import load_rtdetr_coco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="RT-DETR COCO checkpoint (optional)")
    parser.add_argument("--nig", action="store_true", help="E-DETR-NIG (box uncertainty)")
    args = parser.parse_args()

    model = Detector(
        num_classes=80,
        class_head="beta",  # "logit" = RT-DETR
        box_uncertainty=args.nig,
        nig_head_kwargs={"init_sigma": 0.01} if args.nig else None,
        multi_scale=RTDETR_COCO_SCALES,
    )
    if args.weights:
        load_rtdetr_coco(model, args.weights)

    losses = ["evidential_vfl", "boxes"] + (["nig"] if args.nig else [])
    weight_dict = {"loss_evidential_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_nig": 0.01}
    criterion = SetCriterion(80, HungarianMatcher(), weight_dict, losses, alpha=0.75, gamma=2.0)
    postprocessor = PostProcessor(num_top_queries=300, score_type="expectation")

    # one training step on a dummy batch
    images = torch.rand(2, 3, 640, 640)
    targets = {
        "labels": [torch.tensor([0, 17]), torch.tensor([2])],
        "boxes": [torch.tensor([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.3, 0.4]]),
                  torch.tensor([[0.5, 0.5, 0.4, 0.4]])],
    }  # fmt: skip
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    loss_dict = criterion(
        model(images, targets), targets, kl_weight=1e-3
    )  # KL is annealed in practice
    loss = sum(v for v in loss_dict.values() if v.requires_grad)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    print(f"train loss {loss.item():.3f} over {len(loss_dict)} terms")

    # inference: top-300 detections with per-detection Beta expectation / evidence (and box std)
    model.eval()
    with torch.no_grad():
        dets = postprocessor(
            model(images), image_ids=[0, 1], image_sizes=torch.tensor([[480, 640]] * 2)
        )
    d = dets[0]
    print("top detection:", d.labels[0].item(), f"{d.scores[0].item():.3f}", d.boxes[0].tolist())
    print("extras:", {k: tuple(v.shape) for k, v in d.extras.items()})


if __name__ == "__main__":
    main()
