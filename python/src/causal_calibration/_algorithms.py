"""Calibration backends."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable

from ._utils import weighted_mean


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


@dataclass
class StepModel:
    kind: str
    lower_bounds: list[float]
    upper_bounds: list[float]
    values: list[float]

    def predict_many(self, points: list[float]) -> list[float]:
        predictions: list[float] = []
        for point in points:
            index = bisect_right(self.lower_bounds, point) - 1
            if index < 0:
                index = 0
            if index >= len(self.values):
                index = len(self.values) - 1
            predictions.append(self.values[index])
        return predictions

    def mapping_rows(self) -> list[dict[str, float]]:
        return [
            {
                "lower": lower,
                "upper": upper,
                "value": value,
            }
            for lower, upper, value in zip(self.lower_bounds, self.upper_bounds, self.values)
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
class SmoothModel:
    knots_x: list[float]
    knots_y: list[float]
    slopes: list[float]

    def predict_many(self, points: list[float]) -> list[float]:
        if len(self.knots_x) == 1:
            return [self.knots_y[0]] * len(points)
        predictions: list[float] = []
        for point in points:
            if point <= self.knots_x[0]:
                predictions.append(self.knots_y[0])
                continue
            if point >= self.knots_x[-1]:
                predictions.append(self.knots_y[-1])
                continue
            index = bisect_right(self.knots_x, point) - 1
            x0 = self.knots_x[index]
            x1 = self.knots_x[index + 1]
            y0 = self.knots_y[index]
            y1 = self.knots_y[index + 1]
            m0 = self.slopes[index]
            m1 = self.slopes[index + 1]
            h = x1 - x0
            t = (point - x0) / h
            h00 = (2 * t**3) - (3 * t**2) + 1
            h10 = t**3 - (2 * t**2) + t
            h01 = (-2 * t**3) + (3 * t**2)
            h11 = t**3 - t**2
            predictions.append(h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1)
        return predictions

    def mapping_rows(self) -> list[dict[str, float]]:
        return [
            {
                "x": x_value,
                "y": y_value,
                "slope": slope,
            }
            for x_value, y_value, slope in zip(self.knots_x, self.knots_y, self.slopes)
        ]


def _fit_isotonic_blocks(x: list[float], y: list[float], weights: list[float]) -> tuple[list[dict[str, float]], list[float], list[float]]:
    unique_x, unique_y, unique_w = _collapse_sorted_points(x, y, weights)
    blocks: list[dict[str, float]] = []
    for x_value, y_value, weight in zip(unique_x, unique_y, unique_w):
        blocks.append(
            {
                "lower": x_value,
                "upper": x_value,
                "weight": weight,
                "weighted_x": x_value * weight,
                "weighted_y": y_value * weight,
                "value": y_value,
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left["weight"] + right["weight"]
            weighted_x = left["weighted_x"] + right["weighted_x"]
            weighted_y = left["weighted_y"] + right["weighted_y"]
            blocks.append(
                {
                    "lower": left["lower"],
                    "upper": right["upper"],
                    "weight": total_weight,
                    "weighted_x": weighted_x,
                    "weighted_y": weighted_y,
                    "value": weighted_y / total_weight,
                }
            )
    knots_x = [block["weighted_x"] / block["weight"] for block in blocks]
    knots_y = [block["value"] for block in blocks]
    return blocks, knots_x, knots_y


def fit_isotonic(x: list[float], y: list[float], weights: list[float]) -> StepModel:
    blocks, _, _ = _fit_isotonic_blocks(x, y, weights)
    return StepModel(
        kind="isotonic",
        lower_bounds=[block["lower"] for block in blocks],
        upper_bounds=[block["upper"] for block in blocks],
        values=[block["value"] for block in blocks],
    )


def _endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0:
        return 0.0
    if d0 * d1 < 0 and abs(slope) > abs(3.0 * d0):
        return 3.0 * d0
    return slope


def _monotone_cubic_slopes(knots_x: list[float], knots_y: list[float]) -> list[float]:
    n_knots = len(knots_x)
    if n_knots == 1:
        return [0.0]
    h = [knots_x[index + 1] - knots_x[index] for index in range(n_knots - 1)]
    delta = [(knots_y[index + 1] - knots_y[index]) / h[index] for index in range(n_knots - 1)]
    if n_knots == 2:
        return [delta[0], delta[0]]
    slopes = [0.0] * n_knots
    slopes[0] = _endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    for index in range(1, n_knots - 1):
        if delta[index - 1] == 0 or delta[index] == 0 or delta[index - 1] * delta[index] < 0:
            slopes[index] = 0.0
            continue
        w1 = 2 * h[index] + h[index - 1]
        w2 = h[index] + 2 * h[index - 1]
        slopes[index] = (w1 + w2) / ((w1 / delta[index - 1]) + (w2 / delta[index]))
    return slopes


def fit_smooth_isotonic(x: list[float], y: list[float], weights: list[float]) -> SmoothModel:
    _, knots_x, knots_y = _fit_isotonic_blocks(x, y, weights)
    slopes = _monotone_cubic_slopes(knots_x, knots_y)
    return SmoothModel(knots_x=knots_x, knots_y=knots_y, slopes=slopes)


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


def fit_histogram(x: list[float], y: list[float], weights: list[float], *, n_bins: int = 10) -> StepModel:
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
        if index < 0:
            index = 0
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
    return StepModel(
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
) -> StepModel | SmoothModel | LinearModel:
    method_options = method_options or {}
    dispatch: dict[str, Callable[..., object]] = {
        "isotonic": fit_isotonic,
        "smooth_isotonic": fit_smooth_isotonic,
        "linear": fit_linear,
        "histogram": fit_histogram,
    }
    if method not in dispatch:
        raise ValueError(
            "`method` must be one of 'isotonic', 'smooth_isotonic', 'linear', or 'histogram'."
        )
    if method == "histogram":
        return fit_histogram(x, y, weights, n_bins=int(method_options.get("n_bins", 10)))
    return dispatch[method](x, y, weights)  # type: ignore[return-value]
