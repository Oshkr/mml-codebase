"""
Training script for MML-CLIP.

Examples
--------
# Train the full four-modality model:
python train.py \
    --config configs/mml_clip.yaml

# Resume from a checkpoint:
python train.py --config configs/mml_clip.yaml --checkpoint /path/to/weights_best.pth ...
"""

import argparse
import os
import shutil
import sys
import time

import yaml

import torch
import wandb

from setproctitle import setproctitle
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from mmlandmarks.data import MMLDataset, MultimodalCollator, get_transforms
from mmlandmarks.losses import FullyContrastiveLoss, ImageBindLoss
from mmlandmarks.models import MmlCLIP
from mmlandmarks.utils import AverageMeter, TeeLogger, setup_reproducibility


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MML-CLIP or GS-CLIP")
    p.add_argument("--config", default=None,
                   help="Path to a YAML config file (e.g. configs/mml_clip.yaml). "
                        "CLI flags override YAML values.")

    # Model
    p.add_argument("--backbone", default="openai/clip-vit-large-patch14-336",
                   help="HuggingFace model ID for the CLIP backbone")
    p.add_argument("--output_dim", type=int, default=512,
                   help="Projection head output dimensionality")
    p.add_argument("--cache_dir", default=None,
                   help="Local directory for caching HuggingFace model files")

    # Multimodal settings (MML_CLIP only)
    p.add_argument("--modalities", default="GSTC",
                   help="Active modalities: any combination of G, S, T, C")
    p.add_argument("--loss", default="complete", choices=["complete", "imagebind"],
                   help="complete = all C(n,2) pairs; imagebind = ground vs. rest")
    p.add_argument("--text_sampling", default="random", choices=["random", "first"],
                   help="random = random sentence; first = first sentence of the lead section")
    p.add_argument("--last_sat_only", action="store_true",
                   help="Use only the last satellite image per landmark (most recent in CSV list)")
    p.add_argument("--outdoor_only", action="store_true",
                   help="Use outdoor-only ground image subset CSV instead of the full ground CSV")

    # Data
    p.add_argument("--data_root",
                   help="Dataset root containing train/ subdirectory")
    p.add_argument("--img_size", type=int, default=336)
    p.add_argument("--n_val", type=int, default=1024,
                   help="Number of landmarks reserved for validation")
    p.add_argument("--dataset_seed", type=int, default=42)
    p.add_argument("--subset", action="store_true",
                   help="Use only the first 1000 landmarks (for debugging)")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--batch_size_eval", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--clip_grad", type=float, default=100.0,
                   help="Gradient clipping value (0 to disable)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    # Checkpointing
    p.add_argument("--output_dir",
                   help="Directory to save model checkpoints and logs")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a checkpoint to resume training from")

    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="MMLandmarks")
    p.add_argument("--no_verbose", action="store_true")

    # First pass: get --config path without failing on unknown args
    known, _ = p.parse_known_args()
    if known.config is not None:
        with open(known.config) as f:
            cfg = yaml.safe_load(f)
        # Remove keys not recognised by argparse (e.g. stale YAML fields)
        valid_keys = {a.dest for a in p._actions}
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        p.set_defaults(**cfg)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(args, model, dataloader, loss_fn, optimizer, epoch: int):
    model.train()
    losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    step = 1

    bar = tqdm(dataloader, total=len(dataloader)) if not args.no_verbose else dataloader

    for batch in bar:
        ground, satellite, tokens, gps, _ = batch
        ground = ground.to(args.device) if "G" in args.modalities else None
        satellite = satellite.to(args.device) if "S" in args.modalities else None
        tokens = {k: v.to(args.device) for k, v in tokens.items()} if "T" in args.modalities else None
        gps = gps.to(args.device) if "C" in args.modalities else None
        out = model(ground, satellite, tokens, gps)
        loss = loss_fn(out, model.logit_scale.exp())

        losses.update(loss.item())
        loss.backward()

        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

        optimizer.step()
        optimizer.zero_grad()

        if not args.no_verbose:
            bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{losses.avg:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        if args.wandb:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/step": step + (epoch - 1) * len(dataloader),
            })
        step += 1

    return losses.avg


@torch.no_grad()
def validate(args, model, dataloader, loss_fn):
    model.eval()
    losses = []

    for batch in dataloader:
        ground, satellite, tokens, gps, _ = batch
        ground = ground.to(args.device) if "G" in args.modalities else None
        satellite = satellite.to(args.device) if "S" in args.modalities else None
        tokens = {k: v.to(args.device) for k, v in tokens.items()} if "T" in args.modalities else None
        gps = gps.to(args.device) if "C" in args.modalities else None
        out = model(ground, satellite, tokens, gps)
        loss = loss_fn(out, model.logit_scale.exp())

        losses.append(loss.item())

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = time.strftime("%H%M%S")
    outdoor_tag = f"_outdoor_only" if args.outdoor_only else ""

    run_name = (
        f"{run_id}_{args.modalities}"
        f"_text{args.text_sampling}_loss{args.loss}"
        f"{outdoor_tag}"
        f"_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
    )

    # Output directory
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    sys.stdout = TeeLogger(os.path.join(output_dir, "train.log"))
    shutil.copy(__file__, os.path.join(output_dir, "train.py"))

    setup_reproducibility(args.seed)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    img_size = (args.img_size, args.img_size)

    model = MmlCLIP(
        model_name=args.backbone,
        freeze=True,
        cache_dir=args.cache_dir,
        output_dim=args.output_dim,
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)

    model = model.to(args.device)
    print(f"Model: MML_CLIP | Modalities: {args.modalities} | Backbone: {args.backbone} | Device: {args.device}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    sat_train_tf, gnd_train_tf = get_transforms(args.backbone, img_size, split="train")
    sat_val_tf, gnd_val_tf = get_transforms(args.backbone, img_size, split="val")

    tokenizer = CLIPTokenizer.from_pretrained(args.backbone, cache_dir=args.cache_dir)
    collator = MultimodalCollator(tokenizer, max_length=77)

    root_directory = os.path.join(args.data_root,"train")

    train_ds = MMLDataset(
        root=root_directory,
        transform_satellite=sat_train_tf,
        transform_ground=gnd_train_tf,
        text_sampling=args.text_sampling,
        split="train",
        n_val=args.n_val,
        seed=args.dataset_seed,
        subset=args.subset,
        last_sat_only=args.last_sat_only,
        outdoor_only=args.outdoor_only
    )
    val_ds = MMLDataset(
        root=root_directory,
        transform_satellite=sat_val_tf,
        transform_ground=gnd_val_tf,
        text_sampling=args.text_sampling,
        split="val",
        n_val=args.n_val,
        seed=args.dataset_seed,
        subset=args.subset,
        last_sat_only=args.last_sat_only,
        outdoor_only=args.outdoor_only
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size_eval, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collator)

    print(f"Train: {len(train_ds)} landmarks | Val: {len(val_ds)} landmarks")

    # -----------------------------------------------------------------------
    # Loss & optimiser
    # -----------------------------------------------------------------------
    if args.loss == "complete":
        loss_fn = FullyContrastiveLoss(device=args.device)
    else:
        loss_fn = ImageBindLoss(device=args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    torch.cuda.reset_peak_memory_stats(args.device)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'─' * 30} Epoch {epoch} {'─' * 30}")

        train_loss = train_one_epoch(args, model, train_dl, loss_fn, optimizer, epoch)
        val_loss = validate(args, model, val_dl, loss_fn)

        peak_mb = torch.cuda.max_memory_allocated(args.device) / 1024**2
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | Peak VRAM: {peak_mb:.0f} MB")

        if args.wandb:
            wandb.log({"val/loss": val_loss, "val/epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(output_dir, "weights_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved to {ckpt_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":

    setproctitle("MML_training")

    main()
