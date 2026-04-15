"""Shared utilities for the causal calibration package."""

from __future__ import annotations

from statistics import NormalDist
from typing import Iterable, Sequence


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float))


def as_vector(values: Sequence[float] | Iterable[float] | float, name: str) -> list[float]:
    if _is_scalar(values):
        return [float(values)]
    try:
        vector = [float(value) for value in values]  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"`{name}` must be a scalar or one-dimensional iterable.") from exc
    if not vector:
        raise ValueError(f"`{name}` must contain at least one value.")
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


def as_matrix_rows(values: Sequence[Sequence[float]] | Sequence[float] | Iterable[float], name: str) -> list[list[float]]:
    if _is_scalar(values):
        raise TypeError(f"`{name}` must be a vector or a two-dimensional matrix-like object.")
    rows = list(values)  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"`{name}` must not be empty.")
    if rows and _is_scalar(rows[0]):
        return [[float(value)] for value in rows]  # type: ignore[arg-type]
    matrix = [[float(cell) for cell in row] for row in rows]  # type: ignore[arg-type]
    n_cols = len(matrix[0])
    if n_cols == 0:
        raise ValueError(f"`{name}` must have at least one column.")
    if any(len(row) != n_cols for row in matrix):
        raise ValueError(f"`{name}` must be rectangular.")
    return matrix


def validate_binary(values: Sequence[float], name: str) -> None:
    invalid = [value for value in values if value not in (0.0, 1.0)]
    if invalid:
        raise ValueError(f"`{name}` must be binary with values in {{0, 1}}.")


def validate_same_length(length: int, **vectors: Sequence[float]) -> None:
    for name, values in vectors.items():
        if len(values) != length:
            raise ValueError(f"`{name}` must have length {length}, received {len(values)}.")


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


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
        raise ValueError("`n_folds` must be at least 2.")
    return [(index % n_folds) + 1 for index in range(length)]


def normal_quantile(level: float) -> float:
    if level <= 0 or level >= 1:
        raise ValueError("`level` must lie in (0, 1).")
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def mapping_grid(values: Sequence[float], n_points: int = 200) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [minimum]
    step = (maximum - minimum) / float(n_points - 1)
    return [minimum + step * index for index in range(n_points)]
