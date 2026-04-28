"""
Geolocalization evaluation for MML-CLIP.

Given ground or satellite query images, predicts GPS coordinates by matching
image embeddings against a gallery of GPS-encoded coordinates. Reports accuracy
at standard distance thresholds (1 / 25 / 200 / 750 / 2500 km).

The GPS gallery is built from the satellite index coordinates plus the query
landmark coordinates (so all ground-truth locations are always reachable).

Example
--------
python evaluate_geolocalization.py \
    --config eval_configs/geoloc.yaml
"""

import argparse
import json
import os
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmlandmarks.data import MMLandmarksQuerySet, get_transforms
from mmlandmarks.metrics import extract_features
from mmlandmarks.models import MmlCLIP
from mmlandmarks.utils import setup_reproducibility
from mmlandmarks.geoutils import load_gps_tensor, geo_accuracy

def parse_args():
    p = argparse.ArgumentParser(description="Geolocalization evaluation (image → GPS)")

    p.add_argument("--config", default=None,
                   help="Path to a YAML config file (e.g. configs/mml_clip.yaml). "
                        "CLI flags override YAML values.")
    
    p.add_argument("--backbone", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--output_dim", type=int, default=512)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model_number", default=None)

    p.add_argument("--direction", default="G2C", choices=["G2C", "S2C"],
                   help="G2C = ground image → GPS; S2C = satellite image → GPS")

    p.add_argument("--data_root", default=None,
                   help="Dataset root containing query/ and index/ subdirectories")
    p.add_argument("--img_size", type=int, default=336)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)

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
    os.makedirs(os.path.join(args.output_dir, args.direction), exist_ok=True)
    setup_reproducibility(seed=0)

    # Model
    model = MmlCLIP(model_name=args.backbone, freeze=True,
                    cache_dir=args.cache_dir, output_dim=args.output_dim)

    checkpoint_name = None
    for file in os.listdir(args.checkpoint):
        if file.startswith(str(args.model_number)):
            checkpoint_name = file
    
    checkpoint_path = os.path.join(args.checkpoint, checkpoint_name, "weights_best.pth")

    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to(args.device).eval()

    # Data
    val_tf, _ = get_transforms(args.backbone, (args.img_size, args.img_size), split="val")

    query_modality = "ground" if args.direction == "G2C" else "satellite"
    if query_modality == "ground":
        query_folder = os.path.join(args.data_root, "query", "ground")
        query_csv    = os.path.join(args.data_root, "query", "mml_query_ground.csv")
    else:
        query_folder = os.path.join(args.data_root, "query", "satellite")
        query_csv    = os.path.join(args.data_root, "query", "mml_query_satellite.csv")

    query_ds = MMLandmarksQuerySet(data_folder=query_folder, meta_csv=query_csv, transforms=val_tf)
    query_dl = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"Direction: {args.direction} | Queries: {len(query_ds)}")

    # GPS gallery: satellite index coords + query landmark coords
    query_meta = pd.read_csv(os.path.join(args.data_root, "query", "mml_query.csv"))
    index_meta = pd.read_csv(os.path.join(args.data_root, "index", "mml_index_satellite.csv"))
    gps_gallery = load_gps_tensor(pd.concat([index_meta[["lat", "lon"]], query_meta[["lat", "lon"]]],
                                             ignore_index=True))
    print(f"GPS gallery size: {len(gps_gallery)}")

    # Encode GPS gallery
    print("Encoding GPS gallery...")
    with torch.no_grad():
        gallery_gps_feats = model(None, None, None, gps_gallery.to(args.device))[-1]
        if args.normalize_features:
            gallery_gps_feats = F.normalize(gallery_gps_feats, dim=-1)

    # Extract query image features
    print("Extracting query features...")
    query_feats, query_ids, query_idx = extract_features(args, model, query_dl, query_modality)

    ql = query_ids.cpu().numpy()
    img_ql = query_idx.cpu().numpy()
    q_gps_map = query_meta.set_index("landmark_id")[["lat", "lon"]].to_dict("index")

    # Geo retrieval
    retrieved = {"query_id": [], "query_path": [], "lat": [], "lon": [],
                 "pred_lat": [], "pred_lon": [], "top5_coords": []}

    print("Computing GPS predictions...")
    for i in tqdm(range(len(ql))):
        scores = (gallery_gps_feats @ query_feats[i].unsqueeze(-1)).squeeze().cpu().numpy()
        
        ranked = np.argsort(scores)
        ranked = ranked[::-1]

        top_index = ranked[0]
        top_5_index = ranked[:5].copy()

        true_gps = q_gps_map[int(ql[i])]
        pred_coord = gps_gallery[top_index].tolist()
        top5_coords = gps_gallery[top_5_index].tolist()

        retrieved["query_id"].append(int(ql[i]))
        retrieved["query_path"].append(query_ds.images[img_ql[i]])
        retrieved["lat"].append(true_gps["lat"])
        retrieved["lon"].append(true_gps["lon"])
        retrieved["pred_lat"].append(pred_coord[0])
        retrieved["pred_lon"].append(pred_coord[1])
        retrieved["top5_coords"].append(top5_coords)

    # Metrics
    metrics = geo_accuracy(retrieved)
    print(" | ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))

    # Save
    stem = f"MML_CLIP_{args.direction}"
    with open(os.path.join(args.output_dir, args.direction, f"{stem}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    df = pd.DataFrame({k: v for k, v in retrieved.items() if k != "top5_coords"})
    df.to_csv(os.path.join(args.output_dir, args.direction, f"{stem}_predictions.csv"), index=False)

    print(f"Results saved to: {os.path.join(args.output_dir, args.direction)}")


if __name__ == "__main__":
    main()
