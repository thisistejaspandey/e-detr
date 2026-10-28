"""VLM decoder + criterion with random text embeddings (no CLIP weights needed)."""

import pytest
import torch
import torch.nn.functional as F

from edetr.criterion import HungarianMatcher
from edetr.vlm.criterion import VLMSetCriterion
from edetr.vlm.decoder import BetaTransformer
from edetr.vlm.postprocess import VLMPostProcessor


def test_vlm_decoder_and_loss():
    torch.manual_seed(0)
    num_classes, clip_dim = 20, 32
    text = F.normalize(torch.randn(num_classes, clip_dim), dim=-1)
    decoder = BetaTransformer(num_classes=num_classes, clip_dim=clip_dim, num_queries=30,
                              num_denoising=10, num_decoder_layers=2)  # fmt: skip
    feats = [torch.randn(2, 256, s, s) for s in (16, 8, 4)]
    vocab = torch.tensor([1, 3, 5, 7])
    targets = {
        "labels": [torch.tensor([3, 5]), torch.tensor([7])],  # absolute class ids
        "boxes": [
            torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.5, 0.3, 0.4]]),
            torch.tensor([[0.5, 0.5, 0.6, 0.6]]),
        ],
    }
    decoder.train()
    out = decoder(feats, text[vocab], targets)
    assert out["pred_beta"].num_classes == len(vocab)
    out["vocab"] = vocab
    out["vocab_mask"] = torch.tensor([[1.0, 1, 1, 0], [1, 0, 0, 1]])
    weights = {"loss_evidential_vfl": 1, "loss_bbox": 5, "loss_giou": 2}
    crit = VLMSetCriterion(HungarianMatcher(), weights, ["evidential_vfl", "boxes"])
    losses = crit(out, targets, kl_weight=1e-3)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()

    decoder.eval()
    with torch.no_grad():
        out = decoder(feats, text)
    out["vlm_scores"] = torch.rand(2, 30, num_classes)
    dets = VLMPostProcessor(num_classes, num_top_queries=15, novel_class_ids=[0, 1])(
        out, [1, 2], torch.tensor([[100, 200], [100, 200]])
    )
    assert dets[0].scores.shape == (15,)

    # uncertainty-guided fusion: per-entry exponent u = 2/S replaces the fixed alphas
    beta = out["pred_beta"]
    u = beta.uncertainty
    expected = beta.confidence.pow(1 - u) * out["vlm_scores"].pow(u)
    dets = VLMPostProcessor(num_classes, num_top_queries=15, score_type="uncertainty_weighted")(
        out, [1, 2], torch.tensor([[100, 200], [100, 200]])
    )
    assert torch.allclose(dets[0].scores, expected[0].flatten().topk(15).values)


def test_draw_detections():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from edetr.postprocess import Detections
    from edetr.visualize import draw_detections

    det = Detections(
        image_id=0,
        boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        labels=torch.tensor([0, 1]),
        scores=torch.tensor([0.9, 0.1]),
        image_size=torch.tensor([100, 200]),
        extras={
            "expectation": torch.tensor([0.9, 0.5]),
            "concentration": torch.tensor([40.0, 4.0]),
        },
    )
    ax = draw_detections(torch.zeros(100, 200, 3), det, ["a", "b"], score_threshold=0.3)
    assert len(ax.patches) == 1  # only the detection above 0.3
    assert ax.texts[0].get_text() == "a 0.90 u=0.05 a=36.00 b=4.00"
