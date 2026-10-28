"""CLIP text embeddings for the LVIS vocabulary (prompt ensemble, see ``edetr.vlm.model``).

    python tools/build_text_embeddings.py data/lvis/lvis_v1_val.json \
        weights/lvis_text_embeddings_rn50.pt --clip-model RN50

Rows are ordered by zero-indexed label (``category_id - 1``).
"""

import argparse
import json

import torch

from edetr.vlm.model import build_text_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("lvis_ann")
    parser.add_argument("output")
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.lvis_ann) as f:
        cats = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    assert [c["id"] for c in cats] == list(range(1, len(cats) + 1)), "expected ids 1..C"
    emb = build_text_embeddings([c["name"] for c in cats], args.clip_model, device=args.device)
    torch.save({"text_embeddings": emb.cpu(), "clip_model": args.clip_model}, args.output)
    print(f"saved {tuple(emb.shape)} -> {args.output}")


if __name__ == "__main__":
    main()
