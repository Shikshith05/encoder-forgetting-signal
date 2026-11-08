"""Tests for the Part C Shapley split; owner: Shikshith V.

These need only numpy and pandas: no torch, open_clip or GPU.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from src.swap_shapley import evaluate, shapley_split, swap_from_features


def unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Return n random L2-normalised d-dimensional row vectors."""
    vectors = rng.normal(size=(n, d))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_features(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return random (I_p, I_f, T_p, T_f, labels) for 200 images and 5 classes."""
    rng = np.random.default_rng(seed)
    I_p = unit_vectors(rng, 200, 16)
    I_f = unit_vectors(rng, 200, 16)
    T_p = unit_vectors(rng, 5, 16)
    T_f = unit_vectors(rng, 5, 16)
    labels = rng.integers(0, 5, size=200)
    return I_p, I_f, T_p, T_f, labels


def run_swap(I_p, I_f, T_p, T_f, labels):
    """Call swap_from_features with fixed name, ids, scale and threshold."""
    return swap_from_features(
        I_p, I_f, T_p, T_f, labels, "test", list(range(len(labels))), scale=100.0, threshold=0.9
    )


def test_split_sums_to_total_for_random_inputs() -> None:
    """F_V + F_L equals the total drop for 100 random accuracy tables."""
    rng = np.random.default_rng(0)
    for values in rng.uniform(0, 100, size=(100, 4)):
        split = shapley_split(*values)
        assert split["F_V"] + split["F_L"] == pytest.approx(split["total"])


def test_text_frozen_gives_no_text_forgetting() -> None:
    """With an unchanged text encoder, F_L and the interaction are zero."""
    split = shapley_split(A_pp=90.0, A_pf=90.0, A_fp=70.0, A_ff=70.0)
    assert split["F_L"] == 0
    assert split["interaction"] == 0
    assert split["F_V"] == pytest.approx(20.0)


def test_vision_frozen_gives_no_vision_forgetting() -> None:
    """With an unchanged vision encoder, F_V and the interaction are zero."""
    split = shapley_split(A_pp=90.0, A_pf=75.0, A_fp=90.0, A_ff=75.0)
    assert split["F_V"] == 0
    assert split["interaction"] == 0
    assert split["F_L"] == pytest.approx(15.0)


def test_all_equal_gives_zeros_and_nan_d() -> None:
    """Four equal accuracies give zero everywhere and an undefined D."""
    split = shapley_split(80.0, 80.0, 80.0, 80.0)
    assert split["F_V"] == 0
    assert split["F_L"] == 0
    assert split["total"] == 0
    assert split["interaction"] == 0
    assert math.isnan(split["D"])


@pytest.mark.parametrize("A_ff", [80.0, 79.5, 79.0, 85.0])
def test_d_is_nan_when_total_at_most_one(A_ff: float) -> None:
    """D is NaN for a total drop of at most 1 point, including negative drops."""
    assert math.isnan(shapley_split(80.0, 79.0, 79.5, A_ff)["D"])


def test_d_is_bounded_when_total_above_one() -> None:
    """D stays within [-1, 1] whenever it is defined."""
    rng = np.random.default_rng(1)
    checked = 0
    for values in rng.uniform(0, 100, size=(500, 4)):
        split = shapley_split(*values)
        if split["total"] > 1.0:
            assert -1.0 <= split["D"] <= 1.0
            checked += 1
    assert checked > 0


def test_evaluate_on_hand_made_features() -> None:
    """Accuracy, argmax and probabilities are right on a small known case."""
    I = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 1])

    acc, pred, conf, probs = evaluate(I, T, labels, scale=100.0)

    assert acc == pytest.approx(200.0 / 3.0)
    assert pred.tolist() == [0, 1, 0]
    assert probs.shape == (3, 2)
    assert probs.sum(axis=1) == pytest.approx(np.ones(3))
    assert conf == pytest.approx(probs.max(axis=1))


def test_evaluate_is_stable_at_huge_scale() -> None:
    """The softmax stays finite when plain exponentiation would overflow."""
    I = np.array([[1.0, 0.0]])
    T = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, _, _, probs = evaluate(I, T, np.array([0]), scale=1e5)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)


def test_swap_identical_features_has_no_forgetting() -> None:
    """Same features on both sides give equal accuracies and zero splits."""
    I_p, _, T_p, _, labels = make_features()
    row, _ = run_swap(I_p, I_p, T_p, T_p, labels)
    assert row["A_pp"] == row["A_pf"] == row["A_fp"] == row["A_ff"]
    assert row["F_V"] == 0
    assert row["F_L"] == 0


def test_swap_unchanged_text_has_exactly_zero_text_share() -> None:
    """When text features are unchanged, both text shares are exactly zero."""
    I_p, I_f, T_p, _, labels = make_features()
    row, _ = run_swap(I_p, I_f, T_p, T_p.copy(), labels)
    assert row["F_L"] == 0
    assert row["M_F_L"] == 0


def test_swap_margin_split_sums_to_total() -> None:
    """The linear-margin split also adds up to the total margin change."""
    I_p, I_f, T_p, T_f, labels = make_features()
    row, _ = run_swap(I_p, I_f, T_p, T_f, labels)
    assert row["M_F_V"] + row["M_F_L"] == pytest.approx(row["M_total"])


def test_swap_predictions_frame_shape_and_columns() -> None:
    """preds_df has one row per image per hybrid and the agreed columns."""
    I_p, I_f, T_p, T_f, labels = make_features()
    _, preds_df = run_swap(I_p, I_f, T_p, T_f, labels)
    assert len(preds_df) == 4 * len(labels)
    assert list(preds_df.columns) == ["pair", "hybrid", "image_id", "true", "pred", "conf"]
    assert set(preds_df["hybrid"]) == {"pp", "pf", "fp", "ff"}
