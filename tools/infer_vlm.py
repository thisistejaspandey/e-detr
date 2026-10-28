"""Run VLM-EDETR / VLM-NIG on the LVIS val set, save detections and print LVIS AP.

    python tools/infer_vlm.py configs/vlm_edetr_lvis.yaml weights/vlm_edetr_lvis.pth \
        --out outputs/vlm_edetr_lvis_val.json

Detections are LVIS/COCO result dicts (absolute xywh) with per-detection extras: Beta
``expectation`` and ``concentration`` (total evidence S), and ``box_std`` for VLM-NIG.
"""

import argparse
import json
import logging
from pathlib import Path

from accelerate import Accelerator

from edetr.data.lvis import build_lvis_loader
from edetr.vlm.build import build_model, build_postprocessor, load_config
from edetr.vlm.engine import lvis_ap, predict
from edetr.vlm.model import load_vlm_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--out", default=None, help="where to write the detections JSON")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-eval", action="store_true", help="skip the LVIS AP computation")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    accelerator = Accelerator()  # fp32, as the EMA model is evaluated during training
    model = build_model(cfg)
    load_vlm_checkpoint(model, args.checkpoint)
    postprocessor = build_postprocessor(cfg, model.num_classes)
    loader = build_lvis_loader(
        cfg.data.img_root, cfg.data.val_ann, False, args.batch_size, cfg.data.num_workers,
        limit=cfg.data.limit,
    )  # fmt: skip
    model, loader = accelerator.prepare(model, loader)
    preds = predict(model, postprocessor, loader, accelerator)

    if accelerator.is_main_process:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(preds, f)
        if not args.no_eval:
            metrics = lvis_ap(cfg.data.val_ann, preds, loader.dataset.image_ids)
            print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
