"""
Evaluation datasets for MMLandmarks retrieval benchmarks.

Expected directory layout under `data_root`:
    query/
        ground/{a}/{b}/{c}/{id}.jpg
        satellite/{a}/{b}/{c}/{id}.png
        text/{a}/{b}/{c}/{id}.json
        mml_query.csv                   (landmark_id, CommonsCategory, lat, lon)
        mml_query_ground.csv            (landmark_id, images)
        mml_query_satellite.csv         (landmark_id, images)
        mml_query_all_satellite.csv     (landmark_id, images) - (all satellite images, not only latest)
        mml_query_text.csv              (landmark_id, json)
        mml_query_text_sentences.csv    (landmark_id, json) - (first sentences where geo-cues are removed)
    index/
        ground/{a}/{b}/{c}/{id}.jpg
        satellite/{a}/{b}/{c}/{id}.png
        mml_index_ground.csv       (images, gldv2_id)
        mml_index_satellite.csv    (images, lat, lon, year)

The `images` column contains space-separated image IDs.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _load_query_data(folder: str, csv_path: str) -> list:
    """Load query set from CSV. Returns list of (img_path, landmark_id) pairs."""
    df = pd.read_csv(csv_path)
    ext = ".png" if "satellite" in folder else ".jpg"
    result = []
    for _, row in df.iterrows():
        lid = int(row["landmark_id"])
        for img_id in str(row["images"]).split():
            path = os.path.join(folder, img_id[0], img_id[1], img_id[2], img_id + ext)
            result.append((path, lid))
    return result


def _load_index_ground_data(folder: str, csv_path: str) -> list:
    """Load ground gallery from CSV. Returns list of image paths."""
    df = pd.read_csv(csv_path)
    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading ground index"):
        img_id = str(row["images"])
        paths.append(os.path.join(folder, img_id[0], img_id[1], img_id[2], img_id + ".jpg"))
    return paths


def _load_index_satellite_data(folder: str, csv_path: str) -> list:
    """Load satellite gallery from CSV. Returns list of image paths."""
    df = pd.read_csv(csv_path)
    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading satellite index"):
        img_id = str(row["images"])
        paths.append(os.path.join(folder, img_id[0], img_id[1], img_id[2], img_id + ".png"))
    return paths


# ---------------------------------------------------------------------------
# Query dataset
# ---------------------------------------------------------------------------

class MMLandmarksQuerySet(Dataset):
    """
    Query set for G2S or S2G retrieval.

    Returns (image_tensor, landmark_id, dataset_index) per item.
    The `images` attribute maps dataset indices back to file paths for
    qualitative analysis.
    """

    def __init__(self, data_folder: str, meta_csv: str, transforms=None):
        """
        Args:
            data_folder: Path to the query modality directory
                         (e.g., {data_root}/query/ground).
            meta_csv:    Path to the query CSV
                         (e.g., {data_root}/query/mml_query_ground.csv).
            transforms:  Albumentations transform pipeline.
        """
        super().__init__()
        self.transforms = transforms
        self.images: list = []
        self.sample_ids: list = []

        for path, lid in _load_query_data(data_folder, meta_csv):
            self.images.append(path)
            self.sample_ids.append(lid)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        label = self.sample_ids[index]

        if self.transforms is not None:
            img = self.transforms(image=np.array(img))["image"]

        return img, label, index


# ---------------------------------------------------------------------------
# Gallery / index dataset
# ---------------------------------------------------------------------------

class MMLandmarksIndexSet(Dataset):
    """
    Gallery set for retrieval evaluation.

    Consists of:
      - The large index set (label = -1, indicating non-query items).
      - The query positives added back into the gallery (label = landmark_id).

    This mirrors standard retrieval evaluation where the ground-truth items
    are included in the gallery.

    The `images` attribute maps dataset indices back to file paths.
    """

    def __init__(
        self,
        index_folder: str,
        meta_csv: str,
        query_folder: str,
        query_csv: str,
        transforms=None,
    ):
        """
        Args:
            index_folder: Path to the index modality directory
                          (e.g., {data_root}/index/satellite).
            meta_csv:     Path to the index CSV
                          (e.g., {data_root}/index/mml_index_satellite.csv).
            query_folder: Path to the corresponding query modality directory
                          (e.g., {data_root}/query/satellite).
            query_csv:    Path to the query modality CSV for positives
                          (e.g., {data_root}/query/mml_query_satellite.csv).
            transforms:   Albumentations transform pipeline.
        """
        super().__init__()
        self.transforms = transforms
        self.images: list = []
        self.sample_ids: list = []

        # Load the main (large) gallery as non-positives
        if "satellite" in index_folder:
            paths = _load_index_satellite_data(index_folder, meta_csv)
        else:
            paths = _load_index_ground_data(index_folder, meta_csv)

        for path in paths:
            self.images.append(path)
            self.sample_ids.append(-1)

        # Add the query split back as positives
        for path, lid in _load_query_data(query_folder, query_csv):
            self.images.append(path)
            self.sample_ids.append(lid)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        label = self.sample_ids[index]

        if self.transforms is not None:
            img = self.transforms(image=np.array(img))["image"]

        return img, label, index


# ---------------------------------------------------------------------------
# Text query dataset
# ---------------------------------------------------------------------------

def _load_text_data(text_folder: str, csv_path: str, text_sampling: str) -> list:
    """Load text descriptions from CSV. Returns list of (sentence, landmark_id) pairs."""
    df = pd.read_csv(csv_path)
    random.seed(42)
    result = []

    if text_sampling == "no_cues":
        for _, row in df.iterrows():
            lid = int(row["landmark_id"])
            sentence = str(row["sentences"])
            result.append((sentence, lid))
    else:
        for _, row in df.iterrows():
            lid = int(row["landmark_id"])
            json_id = str(row["json"])
            json_path = os.path.join(text_folder, json_id[0], json_id[1], json_id[2], json_id + ".json")
            with open(json_path, "r", encoding="utf-8") as fh:
                sections = json.load(fh)

            if text_sampling == "random":
                section = sections[random.choice(list(sections.keys()))]
                paragraph = random.choice(section.split("\n"))
                sentence = random.choice(paragraph.split(". "))
            elif text_sampling == "first":
                first_section = sections[list(sections.keys())[0]]
                sentence = first_section.split("\n")[0].split(". ")[0]
            else:
                raise ValueError(f"text_sampling must be 'no_cues', 'random' or 'first', got '{text_sampling}'")

            result.append((sentence, lid))
    return result


class MMLandmarksTextQuerySet(Dataset):
    """
    Text query set for text-to-X retrieval evaluation.

    Returns (text_string, landmark_id, dataset_index) per item.
    Use with TextCollator for batched tokenisation.
    """

    def __init__(self, text_folder: str, meta_csv: str, text_sampling: str = "first"):
        """
        Args:
            text_folder: Path to the query text directory
                         (e.g., {data_root}/query/text).
            meta_csv:    Path to the query text CSV
                         (e.g., {data_root}/query/mml_query_text.csv).
            text_sampling: "first", "random" or "no_cues".
        """
        super().__init__()
        self.texts: list = []
        self.sample_ids: list = []
        for sentence, lid in _load_text_data(text_folder, meta_csv, text_sampling):
            self.texts.append(sentence)
            self.sample_ids.append(lid)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.sample_ids[index], index


class TextCollator:
    """Collate function that tokenises a batch of text strings."""

    def __init__(self, tokenizer, max_length: int = 77):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        indices = torch.tensor([item[2] for item in batch])

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return tokens, labels, indices
