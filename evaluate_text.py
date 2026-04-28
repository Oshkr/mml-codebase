"""
Text-to-X retrieval evaluation for MML-CLIP.

Given text descriptions of query landmarks, retrieves matching ground (T2G),
satellite (T2S), or GPS coordinates (T2C) from the respective gallery.

Example
--------
python evaluate_text.py \
    --config configs/mml_clip.yaml \
    --checkpoint /path/to/checkpoints \
    --model_number <run_id> \
    --direction T2S \
    --data_root /path/to/MML \
    --output_dir /path/to/results
"""

import argparse
import json
import math
import os

import numpy as np
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from mmlandmarks.data import (
    MMLandmarksIndexSet, MMLandmarksTextQuerySet, TextCollator, get_transforms,
)
from mmlandmarks.metrics import evaluate_retrieval, extract_features
from mmlandmarks.models import MmlCLIP
from mmlandmarks.utils import setup_reproducibility
from mmlandmarks.geoutils import load_gps_tensor, geo_accuracy

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Text-to-X retrieval evaluation")

    p.add_argument("--config", default=None,
                   help="Path to a YAML config file. CLI flags override YAML values.")

    p.add_argument("--backbone", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--output_dim", type=int, default=512)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="Directory containing model subdirectories (one per run)")
    p.add_argument("--model_number", default=None,
                   help="Run identifier used to select the checkpoint subdirectory")

    p.add_argument("--direction", default="T2S", choices=["T2G", "T2S", "T2C"],
                   help="T2G = text → ground; T2S = text → satellite; T2C = text → GPS coordinates")
    p.add_argument("--text_sampling", default="no_cues", choices=["no_cues", "first", "random"],
                   help="no_cues = first sentence with geo-cues removed; first = first sentence of lead section; random = random sentence")
    p.add_argument("--max_length", type=int, default=77)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    checkpoint_name = None
    for f in os.listdir(args.checkpoint):
        if f.startswith(str(args.model_number)):
            checkpoint_name = f
    checkpoint_path = os.path.join(args.checkpoint, checkpoint_name, "weights_best.pth")
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()

    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.backbone, cache_dir=args.cache_dir)
    text_collator = TextCollator(tokenizer, max_length=args.max_length)

    # Text query dataset (same for all directions)
    text_folder = os.path.join(args.data_root, "query", "text")
    if args.text_sampling == "no_cues":
        text_csv = os.path.join(args.data_root, "query", "mml_query_text_sentences.csv")
    else:
        text_csv = os.path.join(args.data_root, "query", "mml_query_text.csv")
    query_ds = MMLandmarksTextQuerySet(
        text_folder=text_folder, meta_csv=text_csv, text_sampling=args.text_sampling,
    )
    query_dl = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=text_collator)

    stem = f"MML_CLIP_{args.direction}_{args.text_sampling}"
    print(f"Direction: {args.direction} | Text sampling: {args.text_sampling} | Queries: {len(query_ds)}")


    sample_text, sample_id, sample_idx = query_ds[0]
    print(f"Text example [idx={sample_idx}, landmark_id={sample_id}]: {sample_text}")


    # ------------------------------------------------------------------
    # T2C: text → GPS coordinates
    # ------------------------------------------------------------------
    if args.direction == "T2C":
        query_meta = pd.read_csv(os.path.join(args.data_root, "query", "mml_query.csv"))
        index_meta = pd.read_csv(os.path.join(args.data_root, "index", "mml_index_satellite.csv"))
        gps_gallery = load_gps_tensor(
            pd.concat([index_meta[["lat", "lon"]], query_meta[["lat", "lon"]]], ignore_index=True)
        )
        print(f"GPS gallery size: {len(gps_gallery)}")

        print("Encoding GPS gallery...")
        with torch.no_grad():
            gallery_gps_feats = model(None, None, None, gps_gallery.to(args.device))[-1]
            if args.normalize_features:
                gallery_gps_feats = F.normalize(gallery_gps_feats, dim=-1)

        print("Extracting text features...")
        text_feats, text_ids, _ = extract_features(args, model, query_dl, "text")

        ql = text_ids.cpu().numpy()
        q_gps_map = query_meta.set_index("landmark_id")[["lat", "lon"]].to_dict("index")

        retrieved = {"query_id": [], "lat": [], "lon": [],
                     "pred_lat": [], "pred_lon": [], "top5_coords": []}

        print("Computing GPS predictions...")
        for i in tqdm(range(len(ql))):
            scores = (gallery_gps_feats @ text_feats[i].unsqueeze(-1)).squeeze().cpu().numpy()
            ranked = np.argsort(scores)[::-1]

            top_index = ranked[0]
            top_5_index = ranked[:5].copy()

            true_gps = q_gps_map[int(ql[i])]
            pred_coord = gps_gallery[top_index].tolist()
            top5_coords = gps_gallery[top_5_index].tolist()

            retrieved["query_id"].append(int(ql[i]))
            retrieved["lat"].append(true_gps["lat"])
            retrieved["lon"].append(true_gps["lon"])
            retrieved["pred_lat"].append(pred_coord[0])
            retrieved["pred_lon"].append(pred_coord[1])
            retrieved["top5_coords"].append(top5_coords)

        metrics = geo_accuracy(retrieved)
        print(" | ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))

        with open(os.path.join(save_dir, f"{stem}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        df = pd.DataFrame({k: v for k, v in retrieved.items() if k != "top5_coords"})
        df.to_csv(os.path.join(save_dir, f"{stem}_predictions.csv"), index=False)

    # ------------------------------------------------------------------
    # T2G / T2S: text → image retrieval
    # ------------------------------------------------------------------
    else:
        val_tf, _ = get_transforms(args.backbone, (args.img_size, args.img_size), split="val")
        gallery_modality = "ground" if args.direction == "T2G" else "satellite"

        if gallery_modality == "ground":
            query_img_folder = os.path.join(args.data_root, "query", "ground")
            query_img_csv    = os.path.join(args.data_root, "query", "mml_query_ground.csv")
            gallery_folder   = os.path.join(args.data_root, "index", "ground")
            gallery_csv      = os.path.join(args.data_root, "index", "mml_index_ground.csv")
        else:
            query_img_folder = os.path.join(args.data_root, "query", "satellite")
            query_img_csv    = os.path.join(args.data_root, "query", "mml_query_satellite.csv")
            gallery_folder   = os.path.join(args.data_root, "index", "satellite")
            gallery_csv      = os.path.join(args.data_root, "index", "mml_index_satellite.csv")

        gallery_ds = MMLandmarksIndexSet(
            index_folder=gallery_folder, meta_csv=gallery_csv,
            query_folder=query_img_folder, query_csv=query_img_csv,
            transforms=val_tf,
        )
        gallery_dl = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        print(f"Gallery: {len(gallery_ds)}")

        _, metrics, results = evaluate_retrieval(
            config=args, model=model,
            query_loader=query_dl, gallery_loader=gallery_dl,
            query_modality="text", gallery_modality=gallery_modality,
            topk=args.topk,
        )

        with open(os.path.join(save_dir, f"{stem}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if isinstance(results, dict) and "query_id" in results:
            pd.DataFrame(results).to_csv(
                os.path.join(save_dir, f"{stem}_retrieval.csv"), index=False
            )

    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
