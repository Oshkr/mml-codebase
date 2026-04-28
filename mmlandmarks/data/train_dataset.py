"""
Training datasets for MMLandmarks.

Dataset directory layout expected under `root` (the train/ split directory):
    ground/
        {a}/{b}/{c}/{id}.jpg
    satellite/
        {a}/{b}/{c}/{id}.png
    text/
        {a}/{b}/{c}/{id}.json
    mml_train.csv                       (landmark_id, CommonsCategory, lat, lon)
    mml_train_ground.csv                (landmark_id, images)
    mml_train_ground_subset.csv         (landmark_id, images) - (images from outdoors only)
    mml_train_satellite.csv             (landmark_id, images)
    mml_train_text.csv                  (landmark_id, json)

The `images` column contains space-separated image IDs.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _img_path(root: str, modality: str, img_id: str) -> str:
    ext = ".jpg" if modality == "ground" else ".png"
    return os.path.join(root, modality, img_id[0], img_id[1], img_id[2], img_id + ext)

class MMLDataset(Dataset):
    """
    Full four-modality dataset (ground + satellite + text + GPS).

    Args:
        root:            Path to the train/ split directory.
        transform_satellite / transform_ground: Albumentations transforms.
        text_sampling:   How to sample text from a landmark's Wikipedia description.
                         "random" — random sentence from a random section.
                         "first"  — first sentence of the lead section.
        split / n_val / seed / subset: Same as CVDataset.
    """

    def __init__(
        self,
        root: str,
        transform_satellite=None,
        transform_ground=None,
        text_sampling: str = "random",
        split: str = "train",
        n_val: int = 1024,
        seed: int = 42,
        subset: bool = False,
        last_sat_only: bool = False,
        outdoor_only: bool = False,
    ):
        super().__init__()
        self.root = root
        self.text_sampling = text_sampling
        self.last_sat_only = last_sat_only

        meta = pd.read_csv(os.path.join(root, "mml_train.csv"))
        self._lat_lon = meta.set_index("landmark_id")[["lat", "lon"]].to_dict("index")

        ground_csv = "mml_train_ground_subset.csv" if outdoor_only else "mml_train_ground.csv"
        ground_df = pd.read_csv(os.path.join(root, ground_csv))
        sat_df = pd.read_csv(os.path.join(root, "mml_train_satellite.csv"))
        text_df = pd.read_csv(os.path.join(root, "mml_train_text.csv"))

        self._ground_imgs = {
            row["landmark_id"]: row["images"].split()
            for _, row in ground_df.iterrows()
        }
        self._sat_imgs = {
            row["landmark_id"]: row["images"].split()
            for _, row in sat_df.iterrows()
        }
        self._text_ids = {
            row["landmark_id"]: str(row["json"])
            for _, row in text_df.iterrows()
        }

        all_ids = sorted(self._ground_imgs.keys())
        if subset:
            all_ids = all_ids[:1000]
            n_val = 100

        rng = random.Random(seed)
        indices = list(range(len(all_ids)))
        rng.shuffle(indices)

        if split == "train":
            self.ids = [all_ids[i] for i in indices[n_val:]]
        elif split in ("val", "validation"):
            self.ids = [all_ids[i] for i in indices[:n_val]]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.transform_ground = transform_ground
        self.transform_satellite = transform_satellite

    # ------------------------------------------------------------------
    # Text sampling helpers
    # ------------------------------------------------------------------

    def _sample_text(self, sections: dict) -> str:
        if self.text_sampling == "random":
            section = sections[random.choice(list(sections.keys()))]
            paragraph = random.choice(section.split("\n"))
            return random.choice(paragraph.split(". "))
        elif self.text_sampling == "first":
            first_section = sections[list(sections.keys())[0]]
            first_para = first_section.split("\n")[0]
            return first_para.split(". ")[0]
        else:
            raise ValueError(f"text_sampling must be 'random' or 'first', got '{self.text_sampling}'")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        lid = self.ids[index]

        ground_img = Image.open(_img_path(self.root, "ground", random.choice(self._ground_imgs[lid]))).convert("RGB")
        sat_id = self._sat_imgs[lid][-1] if self.last_sat_only else random.choice(self._sat_imgs[lid])
        sat_img = Image.open(_img_path(self.root, "satellite", sat_id)).convert("RGB")

        if self.transform_ground is not None:
            ground_img = self.transform_ground(image=np.array(ground_img))["image"]
        if self.transform_satellite is not None:
            sat_img = self.transform_satellite(image=np.array(sat_img))["image"]

        text_id = self._text_ids[lid]
        text_path = os.path.join(self.root, "text", text_id[0], text_id[1], text_id[2], text_id + ".json")
        with open(text_path, "r", encoding="utf-8") as f:
            sections = json.load(f)
        text = self._sample_text(sections)

        coords = self._lat_lon[lid]
        return ground_img, sat_img, text, coords["lat"], coords["lon"], lid


class MultimodalCollator:
    """
    Custom collate function for MMLDataset.

    Tokenises text strings in each batch and returns a consistent tuple:
        (ground_imgs, satellite_imgs, token_dict, gps_coords, labels)
    """

    def __init__(self, tokenizer, max_length: int = 77):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        ground_imgs = torch.stack([item[0] for item in batch])
        satellite_imgs = torch.stack([item[1] for item in batch])
        texts = [item[2] for item in batch]
        gps = torch.tensor([[item[3], item[4]] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item[5] for item in batch])

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return ground_imgs, satellite_imgs, tokens, gps, labels
