"""Train VLM-EDETR / VLM-NIG on LVIS.

    accelerate launch --num_processes 4 tools/train_vlm.py configs/vlm_edetr_lvis.yaml
    python tools/train_vlm.py configs/vlm_edetr_lvis.yaml train.batch_size=4 data.limit=64

Writes ``last.pth`` (model, EMA, optimizer, scheduler, epoch) and ``ema_epoch<N>.pth`` (EMA
weights) to ``train.output_dir``, and logs LVIS AP of the EMA model every ``eval_every``
epochs.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from edetr.data.lvis import RepeatFactorSampler, build_lvis_loader
from edetr.vlm.build import build_criterion, build_model, build_postprocessor, load_config
from edetr.vlm.engine import ModelEMA, build_optimizer, lvis_ap, predict, train_one_epoch
from edetr.vlm.model import load_vlm_checkpoint

logger = logging.getLogger("train_vlm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    t, d = cfg.train, cfg.data

    accelerator = Accelerator(mixed_precision=t.mixed_precision)
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(t.seed)
    out_dir = Path(t.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")

    model = build_model(cfg)
    if t.init_weights:
        load_vlm_checkpoint(model, t.init_weights)
    criterion = build_criterion(cfg)
    postprocessor = build_postprocessor(cfg, model.num_classes)

    per_gpu = t.batch_size // accelerator.num_processes
    train_loader = build_lvis_loader(
        d.img_root, d.train_ann, True, per_gpu, d.num_workers, d.exclude_frequencies,
        d.repeat_thresh, t.seed, d.limit,
    )  # fmt: skip
    val_loader = build_lvis_loader(
        d.img_root, d.val_ann, False, per_gpu, d.num_workers, limit=d.limit
    )
    sampler = train_loader.sampler
    assert isinstance(sampler, RepeatFactorSampler)

    optimizer = build_optimizer(model, t.lr, t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / t.warmup_steps)
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    ema = ModelEMA(accelerator.unwrap_model(model), t.ema_decay, t.ema_warmups)

    start_epoch = 0
    if t.resume:
        ckpt = torch.load(t.resume, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, t.epochs):
        sampler.set_epoch(epoch)
        train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, accelerator, ema, epoch,
            t.kl_weight, t.kl_anneal_epochs, t.max_norm,
        )  # fmt: skip

        if accelerator.is_main_process:
            torch.save(
                {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                out_dir / "last.pth",
            )
            torch.save({"model": ema.module.state_dict()}, out_dir / f"ema_epoch{epoch}.pth")

        if (epoch + 1) % t.eval_every == 0 or epoch == t.epochs - 1:
            preds = predict(ema.module, postprocessor, val_loader, accelerator)
            if accelerator.is_main_process:
                image_ids = val_loader.dataset.image_ids
                metrics = lvis_ap(d.val_ann, preds, image_ids)
                logger.info(f"epoch {epoch} LVIS: {json.dumps(metrics)}")
                with open(out_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
