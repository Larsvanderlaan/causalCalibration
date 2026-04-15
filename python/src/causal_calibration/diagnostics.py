"""Diagnostics for calibration error estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._algorithms import fit_histogram
from ._algorithms import fit_backend
from ._utils import (
    as_optional_vector,
    as_vector,
    balanced_folds,
    clip_propensity,
    mapping_grid,
    normal_quantile,
    validate_binary,
    validate_same_length,
    weighted_mean,
)
from .core import _dr_pseudo_outcome


def _linear_loo_curve(
    predictions: list[float],
    pseudo_outcome: list[float],
    sample_weight: list[float],
) -> list[float]:
    sum_w = sum(sample_weight)
    sum_x = sum(weight * value for weight, value in zip(sample_weight, predictions))
    sum_y = sum(weight * value for weight, value in zip(sample_weight, pseudo_outcome))
    sum_xx = sum(weight * value * value for weight, value in zip(sample_weight, predictions))
    sum_xy = sum(weight * x_value * y_value for weight, x_value, y_value in zip(sample_weight, predictions, pseudo_outcome))
    loo_curve: list[float] = []
    for weight_i, x_i, y_i in zip(sample_weight, predictions, pseudo_outcome):
        remaining_w = sum_w - weight_i
        if remaining_w <= 0:
            loo_curve.append(y_i)
            continue
        remaining_x = sum_x - (weight_i * x_i)
        remaining_y = sum_y - (weight_i * y_i)
        remaining_xx = sum_xx - (weight_i * x_i * x_i)
        remaining_xy = sum_xy - (weight_i * x_i * y_i)
        mean_x = remaining_x / remaining_w
        mean_y = remaining_y / remaining_w
        denominator = remaining_xx - (remaining_w * mean_x * mean_x)
        if denominator == 0:
            slope = 0.0
        else:
            numerator = remaining_xy - (remaining_w * mean_x * mean_y)
            slope = max(0.0, numerator / denominator)
        intercept = mean_y - (slope * mean_x)
        loo_curve.append(intercept + slope * x_i)
    return loo_curve


def _histogram_loo_curve(
    predictions: list[float],
    pseudo_outcome: list[float],
    sample_weight: list[float],
    method_options: dict[str, float] | None,
) -> list[float]:
    histogram = fit_histogram(
        predictions,
        pseudo_outcome,
        sample_weight,
        n_bins=int((method_options or {}).get("n_bins", 10)),
    )
    bin_ids = []
    for prediction in predictions:
        index = 0
        while index + 1 < len(histogram.lower_bounds) and prediction >= histogram.lower_bounds[index + 1]:
            index += 1
        bin_ids.append(index)
    bin_weight = [0.0] * len(histogram.values)
    bin_sum = [0.0] * len(histogram.values)
    for index, weight, target in zip(bin_ids, sample_weight, pseudo_outcome):
        bin_weight[index] += weight
        bin_sum[index] += weight * target
    overall_weight = sum(sample_weight)
    overall_sum = sum(weight * target for weight, target in zip(sample_weight, pseudo_outcome))
    loo_curve: list[float] = []
    for bin_index, weight_i, target_i in zip(bin_ids, sample_weight, pseudo_outcome):
        remaining_weight = bin_weight[bin_index] - weight_i
        if remaining_weight > 0:
            loo_curve.append((bin_sum[bin_index] - weight_i * target_i) / remaining_weight)
            continue
        if bin_index > 0:
            loo_curve.append(histogram.values[bin_index - 1])
            continue
        loo_curve.append((overall_sum - weight_i * target_i) / max(overall_weight - weight_i, 1e-12))
    return loo_curve


def _estimate_curve(
    predictions: list[float],
    pseudo_outcome: list[float],
    sample_weight: list[float],
    method: str,
    method_options: dict[str, float] | None,
    fold_ids: list[int],
) -> dict[str, Any]:
    model = fit_backend(
        method=method,
        x=predictions,
        y=pseudo_outcome,
        weights=sample_weight,
        method_options=method_options,
    )
    fitted = model.predict_many(predictions)
    if method == "linear":
        oof_curve = _linear_loo_curve(predictions, pseudo_outcome, sample_weight)
    elif method == "histogram":
        oof_curve = _histogram_loo_curve(predictions, pseudo_outcome, sample_weight, method_options)
    else:
        oof_curve = [0.0] * len(predictions)
        for fold in sorted(set(fold_ids)):
            keep = [index for index, fold_id in enumerate(fold_ids) if fold_id != fold]
            holdout = [index for index, fold_id in enumerate(fold_ids) if fold_id == fold]
            model_oof = fit_backend(
                method=method,
                x=[predictions[index] for index in keep],
                y=[pseudo_outcome[index] for index in keep],
                weights=[sample_weight[index] for index in keep],
                method_options=method_options,
            )
            holdout_predictions = model_oof.predict_many([predictions[index] for index in holdout])
            for index, value in zip(holdout, holdout_predictions):
                oof_curve[index] = value
    plugin_terms = [(fitted_value - prediction) ** 2 for fitted_value, prediction in zip(fitted, predictions)]
    robust_terms = [
        ((oof_value - prediction) ** 2) + (2.0 * (oof_value - prediction) * (pseudo - oof_value))
        for oof_value, prediction, pseudo in zip(oof_curve, predictions, pseudo_outcome)
    ]
    return {
        "model": model,
        "fitted": fitted,
        "oof_curve": oof_curve,
        "plugin_estimate": weighted_mean(plugin_terms, sample_weight),
        "estimate": weighted_mean(robust_terms, sample_weight),
    }


@dataclass
class CalibrationDiagnostics:
    """Diagnostics for a calibrated or uncalibrated predictor."""

    estimate: float
    plugin_estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    curve_predictions: list[float]
    curve_estimates: list[float]
    method: str
    n_folds: int
    fold_estimates: list[float] = field(default_factory=list)
    comparison_estimate: float | None = None
    comparison_standard_error: float | None = None

    def summary(self) -> dict[str, Any]:
        summary = {
            "estimate": self.estimate,
            "plugin_estimate": self.plugin_estimate,
            "standard_error": self.standard_error,
            "confidence_interval": self.confidence_interval,
            "curve_method": self.method,
            "jackknife_folds": self.n_folds,
        }
        if self.comparison_estimate is not None:
            summary["comparison_estimate"] = self.comparison_estimate
            summary["comparison_standard_error"] = self.comparison_standard_error
            summary["improvement"] = self.comparison_estimate - self.estimate
        return summary

    def curve_frame(self) -> list[dict[str, float]]:
        return [
            {"prediction": prediction, "estimated_calibration": estimate}
            for prediction, estimate in zip(self.curve_predictions, self.curve_estimates)
        ]

    def plot_data(self) -> dict[str, Any]:
        return {
            "curve": self.curve_frame(),
            "estimate": self.estimate,
            "interval": self.confidence_interval,
        }


def _diagnose_one(
    predictions: list[float],
    treatment: list[float],
    outcome: list[float],
    mu0: list[float],
    mu1: list[float],
    propensity: list[float],
    sample_weight: list[float],
    method: str,
    method_options: dict[str, float] | None,
    fold_ids: list[int],
    level: float,
) -> CalibrationDiagnostics:
    pseudo_outcome = _dr_pseudo_outcome(
        treatment=treatment,
        outcome=outcome,
        mu0=mu0,
        mu1=mu1,
        propensity=propensity,
    )
    full = _estimate_curve(
        predictions=predictions,
        pseudo_outcome=pseudo_outcome,
        sample_weight=sample_weight,
        method=method,
        method_options=method_options,
        fold_ids=fold_ids,
    )
    unique_folds = sorted(set(fold_ids))
    fold_estimates: list[float] = []
    for fold in unique_folds:
        keep = [index for index, fold_id in enumerate(fold_ids) if fold_id != fold]
        subset = _estimate_curve(
            predictions=[predictions[index] for index in keep],
            pseudo_outcome=[pseudo_outcome[index] for index in keep],
            sample_weight=[sample_weight[index] for index in keep],
            method=method,
            method_options=method_options,
            fold_ids=[fold_ids[index] for index in keep],
        )
        fold_estimates.append(subset["estimate"])
    mean_estimate = sum(fold_estimates) / len(fold_estimates)
    variance = ((len(unique_folds) - 1.0) / len(unique_folds)) * sum(
        (estimate - mean_estimate) ** 2 for estimate in fold_estimates
    )
    standard_error = variance ** 0.5
    z_score = normal_quantile(level)
    grid = mapping_grid(predictions)
    grid_estimates = full["model"].predict_many(grid)
    return CalibrationDiagnostics(
        estimate=full["estimate"],
        plugin_estimate=full["plugin_estimate"],
        standard_error=standard_error,
        confidence_interval=(full["estimate"] - z_score * standard_error, full["estimate"] + z_score * standard_error),
        curve_predictions=grid,
        curve_estimates=grid_estimates,
        method=method,
        n_folds=len(unique_folds),
        fold_estimates=fold_estimates,
    )


def diagnose_calibration(
    *,
    predictions: list[float] | tuple[float, ...],
    treatment: list[float] | tuple[float, ...],
    outcome: list[float] | tuple[float, ...],
    mu0: list[float] | tuple[float, ...],
    mu1: list[float] | tuple[float, ...],
    propensity: list[float] | tuple[float, ...],
    sample_weight: list[float] | tuple[float, ...] | None = None,
    comparison_predictions: list[float] | tuple[float, ...] | None = None,
    curve_method: str = "histogram",
    method_options: dict[str, float] | None = None,
    fold_ids: list[int] | tuple[int, ...] | None = None,
    jackknife_folds: int = 100,
    clip: float = 1e-6,
    confidence_level: float = 0.95,
) -> CalibrationDiagnostics:
    prediction_vector = as_vector(predictions, "predictions")
    n_obs = len(prediction_vector)
    treatment_vector = as_vector(treatment, "treatment")
    outcome_vector = as_vector(outcome, "outcome")
    mu0_vector = as_vector(mu0, "mu0")
    mu1_vector = as_vector(mu1, "mu1")
    propensity_vector = clip_propensity(as_vector(propensity, "propensity"), clip)
    sample_weight_vector = as_optional_vector(sample_weight, "sample_weight", n_obs)
    validate_same_length(
        n_obs,
        treatment=treatment_vector,
        outcome=outcome_vector,
        mu0=mu0_vector,
        mu1=mu1_vector,
        propensity=propensity_vector,
        sample_weight=sample_weight_vector,
    )
    validate_binary(treatment_vector, "treatment")
    if fold_ids is None:
        fold_vector = balanced_folds(n_obs, jackknife_folds)
    else:
        fold_vector = [int(value) for value in fold_ids]
        validate_same_length(n_obs, fold_ids=fold_vector)
    diagnostics = _diagnose_one(
        predictions=prediction_vector,
        treatment=treatment_vector,
        outcome=outcome_vector,
        mu0=mu0_vector,
        mu1=mu1_vector,
        propensity=propensity_vector,
        sample_weight=sample_weight_vector,
        method=curve_method,
        method_options=method_options,
        fold_ids=fold_vector,
        level=confidence_level,
    )
    if comparison_predictions is not None:
        comparison_vector = as_vector(comparison_predictions, "comparison_predictions")
        validate_same_length(n_obs, comparison_predictions=comparison_vector)
        comparison = _diagnose_one(
            predictions=comparison_vector,
            treatment=treatment_vector,
            outcome=outcome_vector,
            mu0=mu0_vector,
            mu1=mu1_vector,
            propensity=propensity_vector,
            sample_weight=sample_weight_vector,
            method=curve_method,
            method_options=method_options,
            fold_ids=fold_vector,
            level=confidence_level,
        )
        diagnostics.comparison_estimate = comparison.estimate
        diagnostics.comparison_standard_error = comparison.standard_error
    return diagnostics
