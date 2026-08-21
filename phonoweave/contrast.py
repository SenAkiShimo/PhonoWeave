from __future__ import annotations

import numpy as np


def standardized_effects(
    plain: np.ndarray,
    rounded: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    combined = np.vstack([plain, rounded])
    scale = np.std(combined, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    delta = (np.mean(rounded, axis=0) - np.mean(plain, axis=0)) / scale
    return {name: float(value) for name, value in zip(feature_names, delta, strict=True)}


def standardized_distance(
    plain: np.ndarray,
    rounded: np.ndarray,
    feature_names: tuple[str, ...],
) -> float:
    effects = np.asarray(
        list(standardized_effects(plain, rounded, feature_names).values()),
        dtype=np.float64,
    )
    return float(np.linalg.norm(effects) / np.sqrt(len(effects)))


def balanced_accuracy(labels: np.ndarray, predicted: np.ndarray) -> float:
    scores: list[float] = []
    for label in (0, 1):
        mask = labels == label
        if np.any(mask):
            scores.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(scores)) if scores else 0.0


def nearest_centroid_predict(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    scale = np.std(train, axis=0, ddof=1)
    scale = np.where(scale < 1e-9, 1.0, scale)
    train_norm = train / scale
    test_norm = test / scale
    plain_center = np.mean(train_norm[train_labels == 0], axis=0)
    rounded_center = np.mean(train_norm[train_labels == 1], axis=0)
    d_plain = np.linalg.norm(test_norm - plain_center, axis=1)
    d_rounded = np.linalg.norm(test_norm - rounded_center, axis=1)
    return (d_rounded < d_plain).astype(np.int8)


def loo_balanced_accuracy(plain: np.ndarray, rounded: np.ndarray) -> float:
    combined = np.vstack([plain, rounded])
    labels = np.array([0] * len(plain) + [1] * len(rounded), dtype=np.int8)
    predicted = np.empty_like(labels)

    for index in range(len(combined)):
        train_mask = np.ones(len(combined), dtype=bool)
        train_mask[index] = False
        predicted[index] = nearest_centroid_predict(
            combined[train_mask],
            labels[train_mask],
            combined[index : index + 1],
        )[0]

    return balanced_accuracy(labels, predicted)


def permutation_p(
    plain: np.ndarray,
    rounded: np.ndarray,
    feature_names: tuple[str, ...],
    permutations: int = 1000,
    seed: int = 1731,
) -> float:
    combined = np.vstack([plain, rounded])
    plain_count = len(plain)
    observed = standardized_distance(plain, rounded, feature_names)
    rng = np.random.default_rng(seed)
    exceed = 0

    for _ in range(permutations):
        order = rng.permutation(len(combined))
        perm_plain = combined[order[:plain_count]]
        perm_rounded = combined[order[plain_count:]]
        if standardized_distance(perm_plain, perm_rounded, feature_names) >= observed:
            exceed += 1

    return float((exceed + 1) / (permutations + 1))
