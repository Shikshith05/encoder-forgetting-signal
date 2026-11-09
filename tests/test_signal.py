"""Unit tests for Part A: Pre-fine-tuning Signal (Kishan).

Tests Fréchet distance, Ledoit-Wolf shrinkage, IIMM calculation, zero-shot accuracy,
and signal table generation. Requires only numpy, scipy, scikit-learn, pandas, and matplotlib.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from src.signal_gap import (
    compute_dataset_pair_signals,
    compute_gap,
    compute_iimm_score,
    compute_zeroshot_accuracy,
    fit_gaussian_ledoit_wolf,
    frechet_distance,
    save_heatmap,
)


def unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Generate n random unit-norm vectors in d dimensions."""
    vecs = rng.normal(size=(n, d))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_fit_gaussian_ledoit_wolf():
    """Ledoit-Wolf produces valid mean and symmetric positive definite covariance."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=1.0, scale=2.0, size=(100, 16))
    mu, sigma = fit_gaussian_ledoit_wolf(X)

    assert mu.shape == (16,)
    assert sigma.shape == (16, 16)
    assert np.allclose(sigma, sigma.T)  # Symmetric
    # Eigenvalues should be strictly positive (positive definite)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals > 0)


def test_frechet_distance_identical_is_zero():
    """Distance between identical distributions must be zero."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.diag([0.5, 0.5, 0.5])
    dist = frechet_distance(mu, sigma, mu, sigma)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_compute_gap_self_gap_is_near_zero():
    """Fréchet gap of features with themselves is ~ 0."""
    rng = np.random.default_rng(0)
    features = unit_vectors(rng, 100, 32)
    gap = compute_gap(features, features)
    assert gap == pytest.approx(0.0, abs=1e-5)


def test_frechet_distance_symmetry():
    """Fréchet distance must be symmetric: d(A, B) == d(B, A)."""
    rng = np.random.default_rng(123)
    f1 = unit_vectors(rng, 50, 16)
    f2 = unit_vectors(rng, 50, 16)

    gap12 = compute_gap(f1, f2)
    gap21 = compute_gap(f2, f1)
    assert gap12 == pytest.approx(gap21, abs=1e-6)


def test_zeroshot_accuracy():
    """Zero-shot classification correctly computes accuracy."""
    # 3 classes: standard basis vectors
    T = np.eye(3)
    # 6 images: 2 of class 0, 2 of class 1, 2 of class 2
    I = np.array([
        [0.9, 0.1, 0.0],  # class 0
        [0.8, 0.0, 0.2],  # class 0
        [0.1, 0.9, 0.1],  # class 1
        [0.2, 0.7, 0.1],  # class 1
        [0.0, 0.1, 0.9],  # class 2
        [0.8, 0.1, 0.1],  # class 2 (misclassified as 0!)
    ])
    I = I / np.linalg.norm(I, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, 2, 2])

    acc = compute_zeroshot_accuracy(I, T, labels)
    # 5 out of 6 correct = 83.333%
    assert acc == pytest.approx(5.0 / 6.0 * 100.0, abs=1e-4)


def test_iimm_score():
    """IIMM score components (inter, intra, iimm) calculate properly."""
    rng = np.random.default_rng(42)
    N, C, d = 20, 4, 8
    I = unit_vectors(rng, N, d)
    T = unit_vectors(rng, C, d)
    labels = rng.integers(0, C, size=N)

    scores = compute_iimm_score(I, T, labels)
    assert "inter" in scores
    assert "intra" in scores
    assert "iimm" in scores
    assert scores["iimm"] == pytest.approx((scores["inter"] + scores["intra"]) / 2.0)
    assert -1.0 <= scores["inter"] <= 1.0
    assert -1.0 <= scores["intra"] <= 1.0


def test_compute_dataset_pair_signals():
    """Signal computation produces 12 rows for 4 datasets and correlation."""
    rng = np.random.default_rng(999)
    datasets = ["Pets", "Flowers", "EuroSAT", "DTD"]
    d = 16

    image_dict = {}
    text_dict = {}
    labels_dict = {}

    for ds in datasets:
        image_dict[ds] = unit_vectors(rng, 50, d)
        text_dict[ds] = unit_vectors(rng, 5, d)
        labels_dict[ds] = rng.integers(0, 5, size=50)

    signals_df, gap_v_mat, gap_l_mat, corr = compute_dataset_pair_signals(
        datasets, image_dict, text_dict, labels_dict
    )

    # 4 datasets -> 4 * 3 = 12 ordered pairs
    assert len(signals_df) == 12
    assert gap_v_mat.shape == (4, 4)
    assert gap_l_mat.shape == (4, 4)

    # Diagonal self-gaps should be 0
    for i in range(4):
        assert gap_v_mat[i, i] == 0.0
        assert gap_l_mat[i, i] == 0.0

    # Symmetric matrices
    assert np.allclose(gap_v_mat, gap_v_mat.T)
    assert np.allclose(gap_l_mat, gap_l_mat.T)

    # Correlation should be valid float in [-1, 1]
    assert -1.0 <= corr <= 1.0


def test_save_heatmap():
    """Heatmap file is generated successfully without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "heatmap.png")
        matrix = np.array([[0.0, 1.2], [1.2, 0.0]])
        labels = ["A", "B"]
        save_heatmap(matrix, labels, "Test Heatmap", out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
