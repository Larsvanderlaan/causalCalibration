"""Calibration backends."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from ._utils import mapping_grid, validate_min_unique_scores, weighted_mean

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - dependency guard
    LGBMRegressor = None


_MONOTONE_SPLINE_MAX_INTERNAL_KNOTS = 6
_MONOTONE_SPLINE_DERIVATIVE_DEGREE = 2
_MONOTONE_SPLINE_PENALTY = 1e-3


@dataclass
class HistogramModel:
    kind: str
    lower_bounds: list[float]
    upper_bounds: list[float]
    values: list[float]

    def predict_many(self, points: list[float]) -> list[float]:
        predictions: list[float] = []
        for point in points:
            index = bisect_right(self.lower_bounds, point) - 1
            index = max(0, min(index, len(self.values) - 1))
            predictions.append(self.values[index])
        return predictions

    def mapping_rows(self) -> list[dict[str, float]]:
        return [
            {"lower": lower, "upper": upper, "value": value}
            for lower, upper, value in zip(self.lower_bounds, self.upper_bounds, self.values)
        ]


@dataclass
class LightGBMIsotonicModel:
    estimator: Any
    score_min: float
    score_max: float
    metadata: dict[str, float]

    def predict_many(self, points: list[float]) -> list[float]:
        array = np.asarray(points, dtype=float)
        clipped = np.clip(array, self.score_min, self.score_max).reshape(-1, 1)
        prediction = np.asarray(self.estimator.predict(clipped), dtype=float)
        return prediction.tolist()

    def mapping_rows(self) -> list[dict[str, float]]:
        grid = mapping_grid([self.score_min, self.score_max], n_points=200)
        return [
            {"prediction": point, "calibrated": value}
            for point, value in zip(grid, self.predict_many(grid))
        ]


@dataclass
class LinearModel:
    intercept: float
    slope: float

    def predict_many(self, points: list[float]) -> list[float]:
        return [self.intercept + self.slope * point for point in points]

    def mapping_rows(self) -> list[dict[str, float]]:
        return [{"intercept": self.intercept, "slope": self.slope}]


@dataclass
class MonotoneSplineModel:
    intercept: float
    coef: np.ndarray
    knots: np.ndarray
    derivative_degree: int
    score_min: float
    score_scale: float
    y_min: float
    y_max: float
    metadata: dict[str, float]

    def predict_many(self, points: list[float]) -> list[float]:
        scores = np.asarray(points, dtype=float)
        if np.isclose(self.score_scale, 0.0):
            return np.repeat(self.intercept, len(scores)).tolist()
        scaled = np.clip((scores - self.score_min) / self.score_scale, 0.0, 1.0)
        pred = _evaluate_monotone_spline(
            scaled,
            knots=self.knots,
            coef=self.coef,
            degree=self.derivative_degree,
            intercept=self.intercept,
        )
        pred = np.clip(pred, self.y_min, self.y_max)
        return pred.tolist()

    def mapping_rows(self) -> list[dict[str, float]]:
        grid = mapping_grid([self.score_min, self.score_min + self.score_scale], n_points=200)
        return [
            {"prediction": point, "calibrated": value}
            for point, value in zip(grid, self.predict_many(grid))
        ]


def _collapse_sorted_points(x: list[float], y: list[float], weights: list[float]) -> tuple[list[float], list[float], list[float]]:
    triples = sorted(zip(x, y, weights), key=lambda item: (item[0], item[1]))
    unique_x: list[float] = []
    unique_y: list[float] = []
    unique_w: list[float] = []
    for x_value, y_value, weight in triples:
        if weight <= 0:
            continue
        if unique_x and x_value == unique_x[-1]:
            updated_weight = unique_w[-1] + weight
            unique_y[-1] = (unique_y[-1] * unique_w[-1] + y_value * weight) / updated_weight
            unique_w[-1] = updated_weight
        else:
            unique_x.append(x_value)
            unique_y.append(y_value)
            unique_w.append(weight)
    if not unique_x:
        raise ValueError("At least one observation must have positive weight.")
    return unique_x, unique_y, unique_w


def fit_isotonic(
    x: list[float],
    y: list[float],
    weights: list[float],
    *,
    max_depth: int = 20,
    min_child_samples: int = 10,
    learning_rate: float = 1.0,
    n_estimators: int = 1,
) -> LightGBMIsotonicModel:
    validate_min_unique_scores(x, "isotonic")
    if LGBMRegressor is None:  # pragma: no cover - dependency guard
        raise ImportError("The `lightgbm` package is required for `method=\"isotonic\"`.")
    estimator = LGBMRegressor(
        objective="regression",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_child_samples=int(min_child_samples),
        monotone_constraints=[1],
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        verbosity=-1,
        random_state=0,
    )
    array_x = np.asarray(x, dtype=float).reshape(-1, 1)
    array_y = np.asarray(y, dtype=float)
    array_w = np.asarray(weights, dtype=float)
    estimator.fit(array_x, array_y, sample_weight=array_w)
    return LightGBMIsotonicModel(
        estimator=estimator,
        score_min=float(np.min(array_x)),
        score_max=float(np.max(array_x)),
        metadata={
            "max_depth": float(max_depth),
            "min_child_samples": float(min_child_samples),
            "learning_rate": float(learning_rate),
            "n_estimators": float(n_estimators),
        },
    )


def fit_linear(x: list[float], y: list[float], weights: list[float]) -> LinearModel:
    x_mean = weighted_mean(x, weights)
    y_mean = weighted_mean(y, weights)
    denominator = sum(weight * (x_value - x_mean) ** 2 for x_value, weight in zip(x, weights))
    if denominator == 0:
        return LinearModel(intercept=y_mean, slope=0.0)
    numerator = sum(
        weight * (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value, weight in zip(x, y, weights)
    )
    slope = max(0.0, numerator / denominator)
    intercept = y_mean - (slope * x_mean)
    return LinearModel(intercept=intercept, slope=slope)


def _choose_monotone_spline_knots(
    scores_scaled: np.ndarray,
    *,
    max_internal_knots: int,
    degree: int,
) -> np.ndarray:
    unique_scores = np.unique(scores_scaled)
    max_allowed = max(0, unique_scores.size - degree - 1)
    n_internal = min(max_internal_knots, max_allowed)
    if n_internal <= 0:
        internal = np.array([], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
        internal = np.quantile(scores_scaled, quantiles)
        internal = np.unique(np.asarray(internal, dtype=float))
        internal = internal[(internal > 1e-8) & (internal < 1.0 - 1e-8)]
    return np.concatenate([np.repeat(0.0, degree + 1), internal, np.repeat(1.0, degree + 1)])


def _integrated_bspline_design(
    scores_scaled: np.ndarray,
    *,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    scores_scaled = np.clip(np.asarray(scores_scaled, dtype=float), 0.0, 1.0)
    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid spline specification: expected at least one basis function.")
    columns = []
    for idx in range(n_basis):
        coef = np.zeros(n_basis, dtype=float)
        coef[idx] = 1.0
        derivative_basis = BSpline(knots, coef, degree, extrapolate=False)
        integral_basis = derivative_basis.antiderivative()
        baseline = float(np.asarray(integral_basis(0.0), dtype=float))
        column = np.asarray(integral_basis(scores_scaled), dtype=float) - baseline
        columns.append(column)
    return np.column_stack(columns)


def _evaluate_monotone_spline(
    scores_scaled: np.ndarray,
    *,
    knots: np.ndarray,
    coef: np.ndarray,
    degree: int,
    intercept: float,
) -> np.ndarray:
    scores_scaled = np.clip(np.asarray(scores_scaled, dtype=float), 0.0, 1.0)
    derivative_spline = BSpline(knots, np.asarray(coef, dtype=float), degree, extrapolate=False)
    integrated_spline = derivative_spline.antiderivative()
    baseline = float(np.asarray(integrated_spline(0.0), dtype=float))
    return float(intercept) + np.asarray(integrated_spline(scores_scaled), dtype=float) - baseline


def _second_difference_penalty(n_basis: int) -> np.ndarray:
    if n_basis < 3:
        return np.eye(n_basis, dtype=float)
    penalty = np.zeros((n_basis - 2, n_basis), dtype=float)
    for idx in range(n_basis - 2):
        penalty[idx, idx : idx + 3] = np.array([1.0, -2.0, 1.0], dtype=float)
    return penalty


def fit_monotone_spline(
    x: list[float],
    y: list[float],
    weights: list[float],
    *,
    max_internal_knots: int = _MONOTONE_SPLINE_MAX_INTERNAL_KNOTS,
    basis_degree: int = _MONOTONE_SPLINE_DERIVATIVE_DEGREE + 1,
    penalty: float = _MONOTONE_SPLINE_PENALTY,
) -> MonotoneSplineModel | LinearModel:
    validate_min_unique_scores(x, "monotone_spline")
    scores = np.asarray(x, dtype=float)
    outcome = np.asarray(y, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    y_min = float(np.min(outcome))
    y_max = float(np.max(outcome))

    if np.allclose(scores, scores[0]) or np.isclose(y_max, y_min):
        mean_y = float(np.average(outcome, weights=sample_weight))
        return LinearModel(intercept=mean_y, slope=0.0)

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_scale = float(score_max - score_min)
    if np.isclose(score_scale, 0.0) or np.unique(scores).size < 4:
        return fit_linear(x, y, weights)

    derivative_degree = max(1, int(basis_degree) - 1)
    scores_scaled = np.clip((scores - score_min) / score_scale, 0.0, 1.0)
    knots = _choose_monotone_spline_knots(
        scores_scaled,
        max_internal_knots=int(max_internal_knots),
        degree=derivative_degree,
    )
    basis = _integrated_bspline_design(scores_scaled, knots=knots, degree=derivative_degree)
    design = np.column_stack([np.ones_like(scores_scaled), basis])
    sqrt_weight = np.sqrt(sample_weight)
    weighted_design = design * sqrt_weight[:, None]
    weighted_response = outcome * sqrt_weight

    penalty_block = _second_difference_penalty(basis.shape[1])
    penalty_design = np.column_stack(
        [np.zeros((penalty_block.shape[0], 1), dtype=float), penalty_block]
    )
    augmented_design = np.vstack([weighted_design, np.sqrt(float(penalty)) * penalty_design])
    augmented_response = np.concatenate(
        [weighted_response, np.zeros(penalty_design.shape[0], dtype=float)]
    )

    lower = np.concatenate([[-np.inf], np.zeros(basis.shape[1], dtype=float)])
    upper = np.full(lower.shape, np.inf, dtype=float)
    result = lsq_linear(
        augmented_design,
        augmented_response,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return fit_linear(x, y, weights)

    return MonotoneSplineModel(
        intercept=float(result.x[0]),
        coef=np.asarray(result.x[1:], dtype=float),
        knots=np.asarray(knots, dtype=float),
        derivative_degree=derivative_degree,
        score_min=score_min,
        score_scale=score_scale,
        y_min=y_min,
        y_max=y_max,
        metadata={
            "max_internal_knots": float(max_internal_knots),
            "basis_degree": float(basis_degree),
            "penalty": float(penalty),
        },
    )


def _weighted_quantile_breaks(x: list[float], weights: list[float], n_bins: int) -> list[float]:
    triples = sorted(zip(x, weights), key=lambda item: item[0])
    total_weight = sum(weights)
    targets = [(bin_index / n_bins) * total_weight for bin_index in range(1, n_bins)]
    breaks: list[float] = []
    cumulative = 0.0
    target_index = 0
    for x_value, weight in triples:
        cumulative += weight
        while target_index < len(targets) and cumulative >= targets[target_index]:
            if not breaks or x_value > breaks[-1]:
                breaks.append(x_value)
            target_index += 1
    return breaks


def fit_histogram(x: list[float], y: list[float], weights: list[float], *, n_bins: int = 10) -> HistogramModel:
    if n_bins < 1:
        raise ValueError("`n_bins` must be at least 1.")
    triples = sorted(zip(x, y, weights), key=lambda item: item[0])
    breaks = _weighted_quantile_breaks(x, weights, n_bins)
    lower_bounds = [triples[0][0]]
    lower_bounds.extend(boundary for boundary in breaks if boundary > lower_bounds[0])
    bin_weight = [0.0] * len(lower_bounds)
    bin_sum = [0.0] * len(lower_bounds)
    upper_bounds = [0.0] * len(lower_bounds)
    for x_value, y_value, weight in triples:
        index = bisect_right(lower_bounds, x_value) - 1
        index = max(0, index)
        bin_weight[index] += weight
        bin_sum[index] += weight * y_value
        upper_bounds[index] = x_value
    values: list[float] = []
    for index in range(len(lower_bounds)):
        if bin_weight[index] == 0:
            fill_value = values[-1] if values else weighted_mean(y, weights)
            values.append(fill_value)
            upper_bounds[index] = lower_bounds[index]
        else:
            values.append(bin_sum[index] / bin_weight[index])
    return HistogramModel(
        kind="histogram",
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        values=values,
    )


def fit_backend(
    method: str,
    x: list[float],
    y: list[float],
    weights: list[float],
    *,
    method_options: dict[str, float] | None = None,
) -> HistogramModel | LightGBMIsotonicModel | MonotoneSplineModel | LinearModel:
    method_options = method_options or {}
    if method == "isotonic":
        return fit_isotonic(
            x,
            y,
            weights,
            max_depth=int(method_options.get("max_depth", 20)),
            min_child_samples=int(method_options.get("min_child_samples", 10)),
            learning_rate=float(method_options.get("learning_rate", 1.0)),
            n_estimators=int(method_options.get("n_estimators", 1)),
        )
    if method == "monotone_spline":
        return fit_monotone_spline(
            x,
            y,
            weights,
            max_internal_knots=int(method_options.get("max_internal_knots", 6)),
            basis_degree=int(method_options.get("basis_degree", 3)),
            penalty=float(method_options.get("penalty", 1e-3)),
        )
    if method == "linear":
        return fit_linear(x, y, weights)
    if method == "histogram":
        return fit_histogram(x, y, weights, n_bins=int(method_options.get("n_bins", 10)))
    raise ValueError(
        "`method` must be one of 'isotonic', 'monotone_spline', 'linear', or 'histogram'."
    )
