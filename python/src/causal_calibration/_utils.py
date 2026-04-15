"""Shared utilities for the causal calibration package."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float))


def _check_finite(vector: Sequence[float], name: str) -> None:
    invalid = [value for value in vector if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"`{name}` must contain only finite numeric values.")


def as_vector(values: Sequence[float] | Iterable[float] | float, name: str) -> list[float]:
    if _is_scalar(values):
        vector = [float(values)]
    else:
        try:
            vector = [float(value) for value in values]  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(f"`{name}` must be a scalar or one-dimensional iterable.") from exc
    if not vector:
        raise ValueError(f"`{name}` must contain at least one value.")
    _check_finite(vector, name)
    return vector


def as_optional_vector(
    values: Sequence[float] | Iterable[float] | None,
    name: str,
    length: int,
    default: float = 1.0,
) -> list[float]:
    if values is None:
        return [float(default)] * length
    vector = as_vector(values, name)
    if len(vector) != length:
        raise ValueError(f"`{name}` must have length {length}, received {len(vector)}.")
    return vector


def as_matrix_rows(
    values: Sequence[Sequence[float]] | Sequence[float] | Iterable[float],
    name: str,
) -> list[list[float]]:
    if _is_scalar(values):
        raise TypeError(f"`{name}` must be a vector or a two-dimensional matrix-like object.")
    rows = list(values)  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"`{name}` must not be empty.")
    if rows and _is_scalar(rows[0]):
        matrix = [[float(value)] for value in rows]  # type: ignore[arg-type]
    else:
        matrix = [[float(cell) for cell in row] for row in rows]  # type: ignore[arg-type]
    n_cols = len(matrix[0])
    if n_cols == 0:
        raise ValueError(f"`{name}` must have at least one column.")
    if any(len(row) != n_cols for row in matrix):
        raise ValueError(f"`{name}` must be rectangular.")
    for idx, row in enumerate(matrix):
        _check_finite(row, f"{name}[{idx}]")
    return matrix


def validate_binary(values: Sequence[float], name: str) -> None:
    invalid = [value for value in values if value not in (0.0, 1.0)]
    if invalid:
        raise ValueError(f"`{name}` must be binary with values in {{0, 1}}.")


def validate_same_length(length: int, **vectors: Sequence[float]) -> None:
    for name, values in vectors.items():
        if len(values) != length:
            raise ValueError(f"`{name}` must have length {length}, received {len(values)}.")


def validate_nonnegative_weights(weights: Sequence[float], name: str = "sample_weight") -> None:
    if any(weight < 0 for weight in weights):
        raise ValueError(f"`{name}` must contain only nonnegative values.")
    if sum(weights) <= 0:
        raise ValueError(f"`{name}` must sum to a positive value.")


def validate_method(method: str) -> str:
    valid = {"isotonic", "monotone_spline", "linear", "histogram"}
    if method not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"`method` must be one of {choices}.")
    return method


def validate_min_unique_scores(values: Sequence[float], method: str) -> None:
    n_unique = len({round(value, 12) for value in values})
    if n_unique < 2:
        raise ValueError("Calibration requires at least two distinct prediction values.")
    if method == "monotone_spline" and n_unique < 4:
        raise ValueError("`monotone_spline` requires at least four distinct prediction values.")


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def weighted_effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    if total <= 0 or squared <= 0:
        return 0.0
    return (total * total) / squared


def clip_propensity(values: Sequence[float], clip: float) -> list[float]:
    if clip < 0 or clip >= 0.5:
        raise ValueError("`clip` must be in [0, 0.5).")
    lower = clip
    upper = 1.0 - clip
    clipped = [min(max(value, lower), upper) for value in values]
    if any(value <= 0 or value >= 1 for value in clipped):
        raise ValueError("Propensity scores must lie strictly between 0 and 1 after clipping.")
    return clipped


def order_statistic_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def balanced_folds(length: int, n_folds: int) -> list[int]:
    if n_folds < 2:
        raise ValueError("`jackknife_folds` must be at least 2.")
    return [(index % n_folds) + 1 for index in range(length)]


def validate_fold_ids(length: int, fold_ids: Sequence[int | float]) -> list[int]:
    folds = [int(value) for value in fold_ids]
    if len(folds) != length:
        raise ValueError(f"`fold_ids` must have length {length}, received {len(folds)}.")
    if any(value < 1 for value in folds):
        raise ValueError("`fold_ids` must contain positive integers.")
    observed = sorted(set(folds))
    expected = list(range(1, max(observed) + 1))
    if observed != expected:
        raise ValueError("`fold_ids` must cover folds consecutively starting at 1 with no gaps.")
    for fold in observed:
        if folds.count(fold) == 0:
            raise ValueError("Each fold in `fold_ids` must contain at least one observation.")
    return folds


def validate_oof_alignment(
    predictions: Sequence[float],
    fold_predictions: Sequence[Sequence[float]],
    fold_ids: Sequence[int],
    tolerance: float = 1e-8,
) -> dict[str, float]:
    max_abs_error = 0.0
    for index, fold in enumerate(fold_ids):
        oof_value = float(fold_predictions[index][fold - 1])
        error = abs(float(predictions[index]) - oof_value)
        max_abs_error = max(max_abs_error, error)
        if error > tolerance:
            raise ValueError(
                "Pooled out-of-fold `predictions` do not match `fold_predictions` at the designated `fold_ids`."
            )
    return {"max_abs_error": max_abs_error}


def normal_quantile(level: float) -> float:
    if level <= 0 or level >= 1:
        raise ValueError("`confidence_level` must lie in (0, 1).")
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def mapping_grid(values: Sequence[float], n_points: int = 200) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [minimum]
    step = (maximum - minimum) / float(max(n_points - 1, 1))
    return [minimum + step * index for index in range(n_points)]


def infer_outcome_mean(
    *,
    mu0: Sequence[float] | None,
    mu1: Sequence[float] | None,
    propensity: Sequence[float] | None,
    outcome_mean: Sequence[float] | None,
) -> list[float] | None:
    if outcome_mean is not None:
        return list(outcome_mean)
    if mu0 is None or mu1 is None or propensity is None:
        return None
    return [
        (1.0 - e_value) * mu0_value + e_value * mu1_value
        for mu0_value, mu1_value, e_value in zip(mu0, mu1, propensity)
    ]


def get_column(mapping: Mapping[str, object], key: str, required: bool = True) -> object | None:
    if key in mapping:
        return mapping[key]
    if required:
        raise KeyError(f"Missing required key `{key}`.")
    return None
