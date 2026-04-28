"""
Retrieval evaluation: feature extraction, CMC curves, mAP, and Recall@k.

Usage
-----
    query_feats, query_ids, query_idx = extract_features(config, model, query_loader, "ground")
    gallery_feats, gallery_ids, gallery_idx = extract_features(config, model, gallery_loader, "satellite")
    results = evaluate_retrieval(query_feats, query_ids, gallery_feats, gallery_ids, ranks=[1, 5, 10])
"""

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

_MODALITY_TO_OUTPUT_IDX = {"ground": 0, "satellite": 1, "text": 2, "gps": 3}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(config, model, dataloader, modality: str):
    """
    Extract and optionally L2-normalise embeddings from a dataloader.

    The dataloader must return ``(image_or_text, label, dataset_index)``
    tuples (i.e. MMLandmarksQuerySet / MMLandmarksIndexSet).

    Args:
        config:     Namespace with at least ``device``, ``normalize_features``, ``verbose``.
        model:      The retrieval model (MmlCLIP).
        dataloader: DataLoader wrapping a query or gallery dataset.
        modality:   One of "ground", "satellite", "text".

    Returns:
        features  [N, D] float32 tensor (on GPU)
        labels    [N]    long tensor
        indices   [N]    long tensor (original dataset indices)
    """
    model.eval()
    time.sleep(0.1)  # allow tqdm to initialise cleanly

    out_idx = _MODALITY_TO_OUTPUT_IDX[modality]
    feat_list, ids_list, idx_list = [], [], []

    bar = tqdm(dataloader, total=len(dataloader)) if config.verbose else dataloader

    with torch.no_grad():
        for img, ids, index in bar:
            idx_list.append(index)
            ids_list.append(ids)

            with autocast("cuda"):
                if modality == "ground":
                    out = model(img.to(config.device), None, None, None)
                elif modality == "satellite":
                    out = model(None, img.to(config.device), None, None)
                elif modality == "text":
                    tokens = {k: v.to(config.device) for k, v in img.items()}
                    out = model(None, None, tokens, None)
                else:
                    raise ValueError(f"Unsupported modality: {modality}")

                feat = out[out_idx]
                if config.normalize_features:
                    feat = F.normalize(feat, dim=-1)

            feat_list.append(feat.to(torch.float32))

    if config.verbose:
        bar.close()

    features = torch.cat(feat_list, dim=0)
    labels = torch.cat(ids_list, dim=0).to(config.device)
    indices = torch.cat(idx_list, dim=0).to(config.device)

    return features, labels, indices


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def _recall_at_k(index: np.ndarray, good_index: np.ndarray, junk_index: np.ndarray) -> torch.Tensor:
    """Compute cumulative match characteristic (CMC) for one query."""
    cmc = torch.zeros(len(index), dtype=torch.int32)
    if good_index.size == 0:
        cmc[0] = -1
        return cmc

    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]
    rows_good = np.argwhere(np.isin(index, good_index)).flatten()
    cmc[rows_good[0]:] = 1
    return cmc


def _average_precision(
    index: np.ndarray,
    good_index: np.ndarray,
    junk_index: np.ndarray,
    topk: int = None,
) -> float:
    """Compute average precision for one query."""
    if good_index.size == 0:
        return 0.0

    if topk is not None:
        index = index[:topk]

    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]
    rows_good = np.argwhere(np.isin(index, good_index)).flatten()

    ngood = len(good_index)
    ap = 0.0
    for i, row in enumerate(rows_good[:ngood]):
        precision = (i + 1) / (row + 1)
        old_precision = i / row if row != 0 else 1.0
        ap += (1.0 / ngood) * (old_precision + precision) / 2

    return ap


def _eval_single_query(
    query_feat: torch.Tensor,
    query_label: int,
    gallery_feats: torch.Tensor,
    gallery_labels: np.ndarray,
    topk: int,
):
    scores = (gallery_feats @ query_feat.unsqueeze(-1)).squeeze().cpu().numpy()
    index = np.argsort(scores)[::-1]  # descending

    good_index = np.argwhere(gallery_labels == query_label)
    junk_index = np.argwhere(gallery_labels == -1)

    try:
        pos_scores = scores[good_index]
        pos_first_good = good_index[np.argmax(pos_scores)][0]
    except (ValueError, IndexError):
        pos_first_good = None

    cmc = _recall_at_k(index, good_index, junk_index)
    ap = _average_precision(index, good_index, junk_index, topk=topk)

    return ap, cmc, index, pos_first_good


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    config,
    model,
    query_loader,
    gallery_loader,
    query_modality: str,
    gallery_modality: str,
    ranks: list = None,
    topk: int = 1000,
    cleanup: bool = True,
):
    """
    End-to-end retrieval evaluation.

    Extracts features, computes ranked lists, and reports Recall@k,
    mAP, and median rank.

    Args:
        config:            Config namespace (see extract_features).
        model:             Retrieval model.
        query_loader:      DataLoader for query images.
        gallery_loader:    DataLoader for gallery images.
        query_modality:    "ground" or "satellite".
        gallery_modality:  "ground" or "satellite".
        ranks:             List of k values for Recall@k (default [1, 5, 10]).
        topk:              Gallery cutoff for mAP computation.
        cleanup:           Free GPU memory after evaluation.

    Returns:
        r1        (float)  Recall@1
        metrics   (dict)   All metric values keyed by name
        results   (dict)   Per-query retrieval information
    """
    if ranks is None:
        ranks = [1, 5, 10]

    print("Extracting query features...")
    query_feats, query_ids, query_idx = extract_features(config, model, query_loader, query_modality)

    print("Extracting gallery features...")
    gallery_feats, gallery_ids, gallery_idx = extract_features(config, model, gallery_loader, gallery_modality)

    ql = query_ids.cpu().numpy()
    gl = gallery_ids.cpu().numpy()
    img_ql = query_idx.cpu().numpy()
    img_gl = gallery_idx.cpu().numpy()

    print(f"Queries: {len(ql)} | Gallery: {len(gl)}")

    CMC = torch.zeros(len(gl), dtype=torch.int32)
    ap_total = 0.0
    first_ranks = []
    results = {
        "query_id": [], "query_path": [],
        "top5_ids": [], "top5_paths": [],
        "rank_of_first_correct": [], "path_of_first_correct": [],
    }

    print("Computing ranked lists...")
    for i in tqdm(range(len(ql))):
        ap, cmc, ranked_idx, pos_first = _eval_single_query(
            query_feats[i], ql[i], gallery_feats, gl, topk
        )

        first_correct = int(np.argwhere(gl[ranked_idx] == ql[i])[0])
        first_ranks.append(first_correct + 1)

        results["query_id"].append(ql[i])
        qds = query_loader.dataset
        q_path = qds.texts[img_ql[i]] if hasattr(qds, "texts") else qds.images[img_ql[i]]
        results["query_path"].append(q_path)
        results["top5_ids"].append([gl[j] for j in ranked_idx[:5]])
        results["top5_paths"].append([gallery_loader.dataset.images[img_gl[j]] for j in ranked_idx[:5]])
        results["rank_of_first_correct"].append(first_correct + 1)
        results["path_of_first_correct"].append(
            gallery_loader.dataset.images[img_gl[int(pos_first)]] if pos_first is not None else None
        )

        if cmc[0] == -1:
            continue
        CMC = CMC + cmc
        ap_total += ap

    # Aggregate metrics
    n_queries = len(ql)
    CMC = CMC.float() / n_queries
    mean_ap = ap_total / n_queries * 100
    median_rank = float(np.median(first_ranks))
    metrics = {"Median Rank": median_rank, "mAP": mean_ap}
    for k in ranks:
        metrics[f"Recall@{k}"] = float(CMC[k - 1]) * 100

    metric_str = " | ".join(
        [f"Median Rank: {median_rank:.1f}"]
        + [f"R@{k}: {metrics[f'Recall@{k}']:.2f}" for k in ranks]
        + [f"mAP: {mean_ap:.2f}"]
    )
    print(metric_str)

    if cleanup:
        del query_feats, query_ids, gallery_feats, gallery_ids
        gc.collect()

    return float(CMC[0]) * 100, metrics, results
