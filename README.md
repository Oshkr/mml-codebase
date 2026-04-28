<div align="center">

# 🏢 MMLandmarks: a Cross-View Instance-Level Benchmark for Geo-Spatial Understanding

[![Paper](https://img.shields.io/badge/arXiv-Paper-B3181B.svg)](https://arxiv.org/abs/2512.17492)
[![Dataset](https://img.shields.io/badge/Dataset-Access-4CAF50)](https://archive.compute.dtu.dk/files/public/projects/MMLandmarks)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mmlandmarks.compute.dtu.dk/)

![ALT TEXT](/figures/Figure_1.jpg)

</div>

## Description

MMLandmarks is a multi-modal, instance-level and continental scale dataset of landmarks collect from the United States of America. We address the shortcomings of datasets and specialized models used for geospatial tasks, such as for geolocalization, retrieval and cross-view localization, and instead propose a dataset that enables training and benchmarking models on all these tasks. In fact, for each landmark we collect multiple ground view images, temporal satellite images, as well as the landmarks' GPS coordinates and textual information. MMLandmarks provides data for $18{,}557$ unique landmarks, where each landmark contains all four modalities. With this dataset, we hope to motivate further multimodal research for geospatial tasks, and multimodal learning in general.

## Dataset

Visit [mmlandmarks](https://github.com/Oshkr/mmlandmarks) to download the dataset. 
The page also extensively covers the contents of the dataset as well as its structure.


## Installation

```bash
git clone https://github.com/Oshkr/mmlandmarks
cd mmlandmarks
pip install -e .
pip install -r requirements.txt
```

## Training

```bash
python train.py \
    --model_type MML_CLIP \
    --modalities GSTC \
    --loss complete \
    --text_sampling random \
    --data_root /path/to/dataset/train \
    --output_dir /path/to/checkpoints \
    --epochs 20 \
    --batch_size 512 \
    --wandb
```

## Evaluation

Training saves checkpoints to `<output_dir>/<run_di>_<model_type>_.../weights_best.pth`.  
Pass `--checkpoint <output_dir>` and `--model_number <run_id>` to the evaluation scripts.

### Standard retrieval (G2S and S2G)

```bash
python evaluate_retrieval.py \
    --model_type MML_CLIP \
    --checkpoint /path/to/checkpoints \
    --model_number <run_id> \
    --direction G2S \
    --data_root /path/to/dataset \
    --output_dir /path/to/results
```

```bash
python evaluate_retrieval.py \
    --model_type MML_CLIP \
    --checkpoint /path/to/checkpoints \
    --model_number <run_id> \
    --direction S2G \
    --data_root /path/to/dataset \
    --output_dir /path/to/results
```

### Geolocalization (G2C and S2C)

```bash
python evaluate_geolocalization.py \
    --checkpoint /path/to/checkpoints \
    --model_number <run_id> \
    --direction G2C \
    --data_root /path/to/dataset \
    --output_dir /path/to/results
```

### Text-to-X retrieval (T2G, T2S, T2C)

```bash
python evaluate_text.py \
    --checkpoint /path/to/checkpoints \
    --model_number <run_id> \
    --direction T2S \
    --text_sampling no_cues \
    --data_root /path/to/dataset \
    --output_dir /path/to/results
```

### Reported metrics

- **Recall@1, @5, @10** — fraction of queries where the correct match appears in the top-k results
- **mAP** — mean average precision (computed over the top-1000 gallery items)
- **Median Rank** — median rank of the first correct retrieval

---

## Project structure

```
mmlandmarks_codebase/
├── mmlandmarks/
│   ├── models/
│   │   ├── mml_clip.py          # MmlCLIP and GSCLIP
│   │   ├── encoders.py          # CLIPImageEncoder, CLIPTextEncoder
│   │   └── location_encoder.py  # LocationEncoder (GPS, RFF, EEP)
│   ├── data/
│   │   ├── train_dataset.py     # CVDataset, MMLDataset, MultimodalCollator
│   │   ├── eval_dataset.py      # MMLandmarksQuerySet, MMLandmarksIndexSet,
│   │   │                        # MMLandmarksTextQuerySet, TextCollator
│   │   └── transforms.py        # get_transforms()
│   ├── losses.py                # InfoNCE, FullyContrastiveLoss, ImageBindLoss
│   ├── metrics.py               # extract_features, evaluate_retrieval
│   └── utils.py                 # AverageMeter, setup_reproducibility, TeeLogger
├── train.py                     # Training entry point
├── evaluate_retrieval.py        # Cross-view retrieval evaluation (G2S, S2G)
├── evaluate_geolocalization.py  # Geolocalization evaluation (G2C, S2C)
├── evaluate_text.py             # Text-to-X retrieval evaluation (T2G, T2S, T2C)
├── configs/
│   ├── mml_clip.yaml            # Training config for MML-CLIP
│   └── gs_clip.yaml             # Training config for GS-CLIP
├── eval_configs/
│   ├── crossview.yaml           # Eval config for G2S / S2G
│   ├── geoloc.yaml              # Eval config for G2C / S2C
│   └── text_to_X.yaml           # Eval config for T2G / T2S / T2C
├── requirements.txt
└── setup.py
```

---

## Acknowledgements

The GPS location encoder is based on [GeoCLIP](https://github.com/VicenteVivan/geo-clip) (Vivanco et al., 2023).