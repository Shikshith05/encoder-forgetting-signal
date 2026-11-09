"""Part A (Kishan): Pre-fine-tuning signal for encoder-level forgetting asymmetry.

Computes representation space distances between datasets using pretrained CLIP ViT-B/32:
- Gap_V: Fréchet distance between image embeddings
- Gap_L: Fréchet distance between class text embeddings (with 10 prompt templates)
- Covariance estimation via Ledoit-Wolf shrinkage for numerical stability
- Plain CLIP zero-shot accuracy
- IIMM (Input-Image Multi-Modal) baseline score per dataset
- Pearson correlation between Gap_V and Gap_L over all 12 dataset pairs
- Heatmaps of pairwise Gap_V and Gap_L
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Statistical & Geometric Signal Functions (Numpy / Scipy / Sklearn only)
# ---------------------------------------------------------------------------


def fit_gaussian_ledoit_wolf(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical mean and Ledoit-Wolf regularized covariance.

    Args:
        features: [N, d] array of embedding vectors.

    Returns:
        mu: [d] mean vector.
        sigma: [d, d] regularized covariance matrix.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D array [N, d], got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples to compute covariance, got {features.shape[0]}")

    mu = np.mean(features, axis=0)
    lw = LedoitWolf(assume_centered=False).fit(features)
    sigma = lw.covariance_
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians.

    d_F^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))

    Returns:
        Fréchet distance as a non-negative float.
    """
    diff = mu1 - mu2
    mean_dist_sq = float(np.dot(diff, diff))

    # Compute matrix square root of product sigma1 @ sigma2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        # Numerical imprecision can introduce tiny imaginary parts
        covmean = np.real(covmean)

    tr_covmean = float(np.trace(covmean))
    cov_dist = float(np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

    total_dist_sq = mean_dist_sq + cov_dist
    if total_dist_sq < 0.0:
        # Rounding errors for nearly identical distributions
        total_dist_sq = 0.0

    return float(np.sqrt(total_dist_sq))


def compute_gap(features1: np.ndarray, features2: np.ndarray) -> float:
    """Compute the Fréchet gap between two sets of feature vectors."""
    mu1, sigma1 = fit_gaussian_ledoit_wolf(features1)
    mu2, sigma2 = fit_gaussian_ledoit_wolf(features2)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_zeroshot_accuracy(
    image_features: np.ndarray, text_features: np.ndarray, labels: np.ndarray
) -> float:
    """Compute top-1 zero-shot classification accuracy in percent.

    Args:
        image_features: [N, d] L2-normalized image features.
        text_features: [C, d] L2-normalized class text features.
        labels: [N] true class indices (0 to C-1).

    Returns:
        accuracy in percent (0.0 to 100.0).
    """
    I = np.asarray(image_features, dtype=np.float64)
    T = np.asarray(text_features, dtype=np.float64)
    labels = np.asarray(labels)

    similarity = I @ T.T
    predictions = np.argmax(similarity, axis=1)
    acc = float((predictions == labels).mean() * 100.0)
    return acc


def compute_iimm_score(
    image_features: np.ndarray, text_features: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute the Input-Image Multi-Modal (IIMM) baseline score.

    - inter: average cosine between each image and its incorrect class texts
    - intra: average cosine between all distinct pairs of images
    - IIMM: (inter + intra) / 2

    Args:
        image_features: [N, d] L2-normalized image features.
        text_features: [C, d] L2-normalized class text features.
        labels: [N] true class indices.

    Returns:
        Dict with keys "inter", "intra", "iimm".
    """
    I = np.asarray(image_features, dtype=np.float64)
    T = np.asarray(text_features, dtype=np.float64)
    labels = np.asarray(labels)

    N, d = I.shape
    C, _ = T.shape

    # 1. Inter-modal: mean cosine similarity with wrong class texts
    sim_it = I @ T.T  # [N, C]
    # Mask out true classes
    mask = np.ones((N, C), dtype=bool)
    mask[np.arange(N), labels] = False
    inter = float(sim_it[mask].mean()) if C > 1 else 0.0

    # 2. Intra-modal: mean cosine similarity between distinct image pairs
    # Sum_{i != j} <u_i, u_j> = ||Sum u_i||^2 - Sum ||u_i||^2
    sum_vectors = np.sum(I, axis=0)
    total_norm_sq = float(np.dot(sum_vectors, sum_vectors))
    self_norms_sq = float(np.sum(I * I))
    if N > 1:
        intra = float((total_norm_sq - self_norms_sq) / (N * (N - 1)))
    else:
        intra = 0.0

    iimm = float((inter + intra) / 2.0)
    return {"inter": inter, "intra": intra, "iimm": iimm}


def compute_dataset_pair_signals(
    dataset_names: Sequence[str],
    image_features_dict: Dict[str, np.ndarray],
    text_features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Compute Gap_V, Gap_L, zero-shot acc, and IIMM across all pairs.

    Returns:
        signals_df: 12-row DataFrame for all ordered pairs (source != target).
        gap_v_matrix: [D, D] symmetric matrix of vision gaps.
        gap_l_matrix: [D, D] symmetric matrix of language gaps.
        correlation: Pearson correlation between Gap_V and Gap_L over the 12 pairs.
    """
    n_datasets = len(dataset_names)
    gap_v_matrix = np.zeros((n_datasets, n_datasets), dtype=np.float64)
    gap_l_matrix = np.zeros((n_datasets, n_datasets), dtype=np.float64)

    # Compute per-dataset metrics first (zero-shot, IIMM, and Gaussian fits)
    zeroshot_acc: Dict[str, float] = {}
    iimm_scores: Dict[str, Dict[str, float]] = {}
    image_gaussians: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    text_gaussians: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name in dataset_names:
        I = image_features_dict[name]
        T = text_features_dict[name]
        y = labels_dict[name]

        zeroshot_acc[name] = compute_zeroshot_accuracy(I, T, y)
        iimm_scores[name] = compute_iimm_score(I, T, y)
        image_gaussians[name] = fit_gaussian_ledoit_wolf(I)
        text_gaussians[name] = fit_gaussian_ledoit_wolf(T)

    # Compute pairwise Fréchet distances
    rows: List[Dict[str, Any]] = []
    gap_v_pairs: List[float] = []
    gap_l_pairs: List[float] = []

    for i, src in enumerate(dataset_names):
        for j, tgt in enumerate(dataset_names):
            if i == j:
                # Distance with itself is 0
                gap_v_matrix[i, j] = 0.0
                gap_l_matrix[i, j] = 0.0
                continue

            if j > i:
                mu_v1, sig_v1 = image_gaussians[src]
                mu_v2, sig_v2 = image_gaussians[tgt]
                gap_v = frechet_distance(mu_v1, sig_v1, mu_v2, sig_v2)

                mu_l1, sig_l1 = text_gaussians[src]
                mu_l2, sig_l2 = text_gaussians[tgt]
                gap_l = frechet_distance(mu_l1, sig_l1, mu_l2, sig_l2)

                gap_v_matrix[i, j] = gap_v
                gap_v_matrix[j, i] = gap_v
                gap_l_matrix[i, j] = gap_l
                gap_l_matrix[j, i] = gap_l
            else:
                gap_v = gap_v_matrix[i, j]
                gap_l = gap_l_matrix[i, j]

            gap_v_pairs.append(gap_v)
            gap_l_pairs.append(gap_l)

            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "gap_v": gap_v,
                    "gap_l": gap_l,
                    "source_zeroshot": zeroshot_acc[src],
                    "target_zeroshot": zeroshot_acc[tgt],
                    "source_iimm": iimm_scores[src]["iimm"],
                    "target_iimm": iimm_scores[tgt]["iimm"],
                    "source_iimm_inter": iimm_scores[src]["inter"],
                    "source_iimm_intra": iimm_scores[src]["intra"],
                }
            )

    signals_df = pd.DataFrame(rows)

    # Pearson correlation over the 12 pairs
    corr_matrix = np.corrcoef(gap_v_pairs, gap_l_pairs)
    corr = float(corr_matrix[0, 1])

    return signals_df, gap_v_matrix, gap_l_matrix, corr


def save_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: str,
    cmap: str = "Blues",
) -> None:
    """Render and save a labeled 4x4 heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix, cmap=cmap)

    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=12, pad=12)

    # Annotate numeric values
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            text_color = "white" if val > (matrix.max() / 2.0) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
