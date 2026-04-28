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

Visit the [MMLandmarks](https://github.com/Oshkr/mmlandmarks) dataset codebase to download the full dataset.
The page also extensively covers the contents of the dataset, as well as its structure.


## Installation

```bash
git clone https://github.com/Oshkr/mml-codebase
cd mml-codebase
pip install -e .
pip install -r requirements.txt
```

## Training

Fill the config files with relevant information about:
- `modalities`: Choose any combination of modalities: (G: Ground, S: Satellite, T: Text, C: Coordinates)
- `cache_dir`: local HuggingFace cache dir for pretrained models.
- `data_root`: root directory for the mmlandmarks dataset.
- `output_dir`: directory where the checkpoints and logs should be stored.

Other useful configurations:
- `loss`: choose between complete contrastive loss, or imagebind-style loss.
- `text_sampling`: choose sampling the first sentence or random sentences.
- `last_sat_only`: if true, always uses the latest satellite from the landmark.
- `outdoor_only`: if true, uses the subset of outdoor filtered ground images.

```bash
python train.py \
    --config configs/mml_clip.yaml \
```

## Evaluation

Training saves checkpoints to `<output_dir>/<run_id>_<model_type>_.../weights_best.pth`.

Fill the config files with relevant information about:
- `cache_dir`: local HuggingFace cache dir for pretrained models.
- `checkpoint`: directory of the model checkpoint.
- `model_number`: `run_id` of the model checkpoint.
- `data_root`: root directory for the mmlandmarks dataset.
- `output_dir`: directory where the results should be stored.

### 🗽Cross-View Localization (G2S and S2G)

- `G2S`:
    - Takes the $18{,}668$ ground images from the $1000$ query landmarks and performs retrieval from the satellite index set, composed of the $100k$ distractor index set + the $1000$ positive satellite matches from the query landmarks.

- `S2G`:
    - Takes the $1000$ satellite images from the $1000$ query landmarks and performs retrieval from the ground index set, composed of the $714k$ distractor index set + the $18{,}668$ positive grund matches from the query landmarks.


```bash
python evaluate_retrieval.py \
    --config eval_configs/crossview.yaml
```

### 🌎 Geolocalization (G2C and S2C)

- `G2C`: 
    - Takes the $18{,}668$ ground images from the $1000$ query landmarks and performs geolocalization by finding the closest GPS coordinates from the satellite index set + the $1000$ GPS coordinates from the query landmarks.
- `S2C`: 
    - Takes the $1000$ satellite images from the $1000$ query landmarks and performs geolocalization by finding the closest GPS coordinates from the satellite index set + the $1000$ GPS coordinates from the query landmarks.
```bash
python evaluate_geolocalization.py \
    --config eval_configs/geoloc.yaml
```

### ✏️ Text-to-X retrieval (T2G, T2S, T2C)

We try retrieving with text in three ways: with the `first` sentences, `random` sentences, or `no_cue` sentences where geographical cues have been removed from the first sentence.  

- `T2G`: 
    - Takes the $1000$ sentences from the $1000$ query landmarks and performs retrieval from the ground index set, composed of the $714k$ distractor index set + the $18{,}668$ positive grund matches from the query landmarks.
- `T2S`:
    - Takes the $1000$ sentences from the $1000$ query landmarks and performs retrieval from the satellite index set, composed of the $100k$ distractor index set + the $1000$ positive satellite matches from the query landmarks.

- `T2C`:
    - Takes the $1000$ sentences from the $1000$ query landmarks and performs geolocalization by finding the closest GPS coordinates from the satellite index set + the $1000$ GPS coordinates from the query landmarks.

```bash
python evaluate_text.py \
    --config eval_configs/text_to_X.yaml
```

## Reported metrics

### Retrieval / Cross-View Localization
- **Recall@1, @5, @10** — fraction of queries where the correct match appears in the top-k results
- **mAP** — mean average precision (computed over the top-1000 gallery items)
- **Median Rank** — median rank of the first correct retrieval

### Geolocalization:
- **Distance (% @ km)** — percentage accuracy at standard distance thresholds (1 / 25 / 200 / 750 / 2500 km).

### 🤖 pretrained model:

We provide a [pretrained model]() trained with the following setup:
- `modalities`: GSTC
- `outdoor_only`: false
- `text_sampling`: first
- `last_sat_only`: false
- `loss`: complete

#### Results:
|  | MedR | mAP@1k | R@1 | R@5 | R@10 |
|:-----:|:---------:|:-------------:|:----------------:|:---------------:|:-----------------:|
| `G2S` | 57 | 24.88 | 19.40 | 43.77 | 56.45 |
| `S2G` | 29 | 16.86 | 27.30 | 48.40 | 57.70 |
| `T2G` | 77 | 16.31 | 28.10 | 44.60 | 52.40 |
| `T2S` | 712 | 17.6 | 13.80 | 31.70 | 40.80 |

|  | Dist @ 1km | Dist @ 25km | Dist @ 200km | Dist @ 750km | Dist @ 2500km |
|:-----:|:---------:|:-------------:|:----------------:|:---------------:|:-----------------:|
| `G2C` | 16.6 | 36.0 | 50.77 | 74.6 | 92.3 |
| `S2C` | 26.4 | 50.3 | 71.3 | 91.9 | 98.9 |
| `T2C`(`no_cues`) | 6.1 | 17.8 | 28.7 | 51.9 | 87.7 |


---

## Acknowledgements

The GPS location encoder is based on [GeoCLIP](https://github.com/VicenteVivan/geo-clip) (Vivanco et al., 2023).

## Citation

If you find this work useful for your research, please consider citing our work as follows:
```
@InProceedings{Kristoffersen_2026_MMLandmarks,
  author    = {Oskar Kristoffersen and Alba Reinders and Morten R. Hannemose and Anders B. Dahl and Dim P. Papadopoulos},
  title     = {MMLandmarks: a Cross-View Instance-Level Benchmark for Geo-Spatial Understanding},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
}
```