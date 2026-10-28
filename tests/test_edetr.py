import torch

from edetr.criterion import HungarianMatcher, SetCriterion
from edetr.detector import Detector
from edetr.heads import BetaHead
from edetr.postprocess import PostProcessor

TARGETS = {
    "labels": [torch.tensor([0, 2]), torch.tensor([1])],
    "boxes": [
        torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.5, 0.3, 0.4]]),
        torch.tensor([[0.5, 0.5, 0.6, 0.6]]),
    ],
}
WEIGHTS = {
    "loss_vfl": 1.0,
    "loss_evidential_vfl": 1.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_nig": 0.01,
}


def _check(class_head, losses, nig=False):
    torch.manual_seed(0)
    model = Detector(num_classes=3, class_head=class_head, box_uncertainty=nig, depth=18)
    crit = SetCriterion(3, HungarianMatcher(), WEIGHTS, losses)
    x = torch.rand(2, 3, 128, 128)
    model.train()
    out = crit(model(x, TARGETS), TARGETS, kl_weight=1e-3)
    loss = sum(v for v in out.values() if v.requires_grad)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    with torch.no_grad():
        dets = PostProcessor(10)(model(x), [1, 2], torch.tensor([[100, 200], [100, 200]]))
    assert len(dets) == 2 and dets[0].boxes.shape == (10, 4)
    assert len(dets[0].to_coco()) == 10
    return dets[0]


def test_rtdetr():
    _check("logit", ["vfl", "boxes"])


def test_edetr():
    d = _check("beta", ["evidential_vfl", "boxes"])
    assert set(d.extras) == {"expectation", "concentration"}


def test_edetr_nig():
    d = _check("beta", ["evidential_vfl", "boxes", "nig"], nig=True)
    assert d.extras["box_std"].shape == (10, 4)


def test_beta_init_from_logits():
    head = BetaHead(8, 5)
    w, b = torch.randn(5, 8), torch.randn(5)
    head.init_from_logits(w, b, concentration=1e6)
    x = torch.randn(4, 8)
    expectation = head.wrap(head(x)).expectation
    assert torch.allclose(expectation, torch.sigmoid(x @ w.t() + b), atol=1e-3)
