"""
Cross-view image retrieval evaluation for MML-CLIP.

Supports G2S (ground → satellite) and S2G (satellite → ground).

Example
--------
python evaluate_retrieval.py \
    --config configs/mml_clip.yaml \
    --checkpoint /path/to/checkpoints \
    --model_number <run_id> \
    --direction G2S \
    --data_root /path/to/MML \
    --output_dir /path/to/results
"""

import argparse
import json
import os
import yaml
import torch

from torch.utils.data import DataLoader

from mmlandmarks.data import MMLandmarksIndexSet, MMLandmarksQuerySet, get_transforms
from mmlandmarks.metrics import evaluate_retrieval
from mmlandmarks.models import MmlCLIP
from mmlandmarks.utils import setup_reproducibility


def parse_args():
    p = argparse.ArgumentParser(description="Cross-view retrieval evaluation")

    p.add_argument("--config", default=None,
                   help="Path to a YAML config file. CLI flags override YAML values.")

    p.add_argument("--backbone", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--output_dim", type=int, default=512)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Directory containing model subdirectories (one per run)")
    p.add_argument("--model_number", default=None,
                   help="Run identifier used to select the checkpoint subdirectory")

    p.add_argument("--direction", default="G2S", choices=["G2S", "S2G"],
                   help="G2S = ground query → satellite gallery; S2G = satellite → ground")

    p.add_argument("--data_root", default=None,
                   help="Dataset root containing query/ and index/ subdirectories")
    p.add_argument("--img_size", type=int, default=336)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=1000)

    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_verbose", action="store_true")
    p.add_argument("--normalize_features", action="store_true", default=True)

    known, _ = p.parse_known_args()
    if known.config is not None:
        with open(known.config) as f:
            cfg = yaml.safe_load(f)
        valid_keys = {a.dest for a in p._actions}
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        p.set_defaults(**cfg)

    return p.parse_args()


def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.verbose = not args.no_verbose

    save_dir = os.path.join(args.output_dir, args.direction)
    os.makedirs(save_dir, exist_ok=True)
    setup_reproducibility(seed=0)

    # Model
    model = MmlCLIP(model_name=args.backbone, freeze=True,
                    cache_dir=args.cache_dir, output_dim=args.output_dim)

    checkpoint_name = next(f for f in os.listdir(args.checkpoint)
                           if f.startswith(str(args.model_number)))
    checkpoint_path = os.path.join(args.checkpoint, checkpoint_name, "weights_best.pth")
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()

    # Data
    val_tf, _ = get_transforms(args.backbone, (args.img_size, args.img_size), split="val")

    if args.direction == "G2S":
        query_modality, gallery_modality = "ground", "satellite"
        query_folder     = os.path.join(args.data_root, "query", "ground")
        query_csv        = os.path.join(args.data_root, "query", "mml_query_ground.csv")
        gt_folder        = os.path.join(args.data_root, "query", "satellite")
        gt_csv           = os.path.join(args.data_root, "query", "mml_query_satellite.csv")
        gallery_folder   = os.path.join(args.data_root, "index", "satellite")
        gallery_meta_csv = os.path.join(args.data_root, "index", "mml_index_satellite.csv")
    else:  # S2G
        query_modality, gallery_modality = "satellite", "ground"
        query_folder     = os.path.join(args.data_root, "query", "satellite")
        query_csv        = os.path.join(args.data_root, "query", "mml_query_satellite.csv")
        gt_folder        = os.path.join(args.data_root, "query", "ground")
        gt_csv           = os.path.join(args.data_root, "query", "mml_query_ground.csv")
        gallery_folder   = os.path.join(args.data_root, "index", "ground")
        gallery_meta_csv = os.path.join(args.data_root, "index", "mml_index_ground.csv")

    query_ds = MMLandmarksQuerySet(data_folder=query_folder, meta_csv=query_csv, transforms=val_tf)
    gallery_ds = MMLandmarksIndexSet(
        index_folder=gallery_folder, meta_csv=gallery_meta_csv,
        query_folder=gt_folder, query_csv=gt_csv, transforms=val_tf,
    )

    query_dl = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    gallery_dl = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Direction: {args.direction} | Queries: {len(query_ds)} | Gallery: {len(gallery_ds)}")

    # Evaluation
    _, metrics, results = evaluate_retrieval(
        config=args, model=model,
        query_loader=query_dl, gallery_loader=gallery_dl,
        query_modality=query_modality, gallery_modality=gallery_modality,
        topk=args.topk,
    )

    # Save
    stem = f"MML_CLIP_{args.direction}"
    with open(os.path.join(save_dir, f"{stem}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd
    if isinstance(results, dict) and "query_id" in results:
        pd.DataFrame(results).to_csv(
            os.path.join(save_dir, f"{stem}_retrieval.csv"), index=False
        )

    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
