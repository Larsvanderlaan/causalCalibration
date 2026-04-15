"""Diagnostics for calibration error estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from warnings import warn

from ._algorithms import fit_backend, fit_histogram
from ._utils import (
    as_optional_vector,
    as_vector,
    balanced_folds,
    clip_propensity,
    infer_outcome_mean,
    mapping_grid,
    normal_quantile,
    validate_binary,
    validate_fold_ids,
    validate_method,
    validate_nonnegative_weights,
    validate_same_length,
    weighted_mean,
)
from .core import _dr_pseudo_outcome, _r_pseudo_target
from .overlap import OverlapDiagnostics, assess_overlap


def _linear_loo_curve(
    predictions: list[float],
    pseudo_outcome: list[float],
    sample_weight: list[float],
) -> list[float]:
    sum_w = sum(sample_weight)
    sum_x = sum(weight * value for weight, value in zip(sample_weight, predictions))
    sum_y = sum(weight * value for weight, value in zip(sample_weight, pseudo_outcome))
    sum_xx = sum(weight * value * value for weight, value in zip(sample_weight, predictions))
    sum_xy = sum(
        weight * x_value * y_value
        for weight, x_value, y_value in zip(sample_weight, predictions, pseudo_outcome)
    )
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
        (pseudo - prediction) * (oof_value - prediction)
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
class CalibrationTargetResult:
    target_population: str
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
            "target_population": self.target_population,
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
    target_population: str
    overlap_diagnostics: OverlapDiagnostics | None = None
    fold_estimates: list[float] = field(default_factory=list)
    comparison_estimate: float | None = None
    comparison_standard_error: float | None = None
    dr_result: CalibrationTargetResult | None = None
    overlap_result: CalibrationTargetResult | None = None

    def summary(self) -> dict[str, Any]:
        summary = {
            "target_population": self.target_population,
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
        if self.overlap_diagnostics is not None:
            summary["overlap"] = self.overlap_diagnostics.summary()
        if self.target_population == "both":
            summary["dr_result"] = self.dr_result.summary() if self.dr_result is not None else None
            summary["overlap_result"] = self.overlap_result.summary() if self.overlap_result is not None else None
        return summary

    def curve_frame(self) -> list[dict[str, float]]:
        return [
            {"prediction": prediction, "estimated_calibration": estimate}
            for prediction, estimate in zip(self.curve_predictions, self.curve_estimates)
        ]

    def plot_data(self) -> dict[str, Any]:
        payload = {
            "curve": self.curve_frame(),
            "estimate": self.estimate,
            "interval": self.confidence_interval,
            "target_population": self.target_population,
        }
        if self.target_population == "both":
            payload["dr_curve"] = [] if self.dr_result is None else self.dr_result.curve_frame()
            payload["overlap_curve"] = [] if self.overlap_result is None else self.overlap_result.curve_frame()
        return payload

    def plot(self, ax: Any | None = None) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Plotting diagnostics requires matplotlib.") from exc

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.curve_predictions, self.curve_estimates, label=self.target_population)
        if self.target_population == "both":
            if self.dr_result is not None:
                ax.plot(
                    self.dr_result.curve_predictions,
                    self.dr_result.curve_estimates,
                    label="dr",
                )
            if self.overlap_result is not None:
                ax.plot(
                    self.overlap_result.curve_predictions,
                    self.overlap_result.curve_estimates,
                    label="overlap",
                )
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Estimated calibration curve")
        ax.legend()
        return ax


def _build_target_result(
    *,
    predictions: list[float],
    pseudo_outcome: list[float],
    sample_weight: list[float],
    method: str,
    method_options: dict[str, float] | None,
    fold_ids: list[int],
    level: float,
    target_population: str,
    comparison_predictions: list[float] | None = None,
) -> CalibrationTargetResult:
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
    result = CalibrationTargetResult(
        target_population=target_population,
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
    if comparison_predictions is not None:
        comparison = _estimate_curve(
            predictions=comparison_predictions,
            pseudo_outcome=pseudo_outcome,
            sample_weight=sample_weight,
            method=method,
            method_options=method_options,
            fold_ids=fold_ids,
        )
        comparison_fold_estimates: list[float] = []
        for fold in unique_folds:
            keep = [index for index, fold_id in enumerate(fold_ids) if fold_id != fold]
            subset = _estimate_curve(
                predictions=[comparison_predictions[index] for index in keep],
                pseudo_outcome=[pseudo_outcome[index] for index in keep],
                sample_weight=[sample_weight[index] for index in keep],
                method=method,
                method_options=method_options,
                fold_ids=[fold_ids[index] for index in keep],
            )
            comparison_fold_estimates.append(subset["estimate"])
        mean_comp = sum(comparison_fold_estimates) / len(comparison_fold_estimates)
        variance_comp = ((len(unique_folds) - 1.0) / len(unique_folds)) * sum(
            (estimate - mean_comp) ** 2 for estimate in comparison_fold_estimates
        )
        result.comparison_estimate = comparison["estimate"]
        result.comparison_standard_error = variance_comp ** 0.5
    return result


def diagnose_calibration(
    *,
    predictions: list[float] | tuple[float, ...],
    treatment: list[float] | tuple[float, ...],
    outcome: list[float] | tuple[float, ...],
    mu0: list[float] | tuple[float, ...],
    mu1: list[float] | tuple[float, ...],
    propensity: list[float] | tuple[float, ...],
    outcome_mean: list[float] | tuple[float, ...] | None = None,
    sample_weight: list[float] | tuple[float, ...] | None = None,
    comparison_predictions: list[float] | tuple[float, ...] | None = None,
    curve_method: str = "histogram",
    method_options: dict[str, float] | None = None,
    fold_ids: list[int] | tuple[int, ...] | None = None,
    jackknife_folds: int = 100,
    clip: float = 1e-6,
    confidence_level: float = 0.95,
    target_population: str = "dr",
) -> CalibrationDiagnostics:
    if target_population not in {"dr", "overlap", "both"}:
        raise ValueError("`target_population` must be one of 'dr', 'overlap', or 'both'.")
    method_name = validate_method(curve_method)
    prediction_vector = as_vector(predictions, "predictions")
    n_obs = len(prediction_vector)
    treatment_vector = as_vector(treatment, "treatment")
    outcome_vector = as_vector(outcome, "outcome")
    mu0_vector = as_vector(mu0, "mu0")
    mu1_vector = as_vector(mu1, "mu1")
    propensity_raw = as_vector(propensity, "propensity")
    propensity_vector = clip_propensity(propensity_raw, clip)
    sample_weight_vector = as_optional_vector(sample_weight, "sample_weight", n_obs)
    validate_nonnegative_weights(sample_weight_vector)
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
    fold_vector = balanced_folds(n_obs, jackknife_folds) if fold_ids is None else validate_fold_ids(n_obs, fold_ids)
    overlap = assess_overlap(
        treatment=treatment_vector,
        propensity=propensity_raw,
        sample_weight=sample_weight_vector,
        clip=clip,
    )
    for message in overlap.recommendation_messages():
        warn(message, stacklevel=2)

    comparison_vector = None if comparison_predictions is None else as_vector(comparison_predictions, "comparison_predictions")
    if comparison_vector is not None:
        validate_same_length(n_obs, comparison_predictions=comparison_vector)

    dr_result = None
    overlap_result = None
    if target_population in {"dr", "both"}:
        dr_result = _build_target_result(
            predictions=prediction_vector,
            pseudo_outcome=_dr_pseudo_outcome(
                treatment=treatment_vector,
                outcome=outcome_vector,
                mu0=mu0_vector,
                mu1=mu1_vector,
                propensity=propensity_vector,
            ),
            sample_weight=sample_weight_vector,
            method=method_name,
            method_options=method_options,
            fold_ids=fold_vector,
            level=confidence_level,
            target_population="dr",
            comparison_predictions=comparison_vector,
        )

    if target_population in {"overlap", "both"}:
        outcome_mean_vector = infer_outcome_mean(
            mu0=mu0_vector,
            mu1=mu1_vector,
            propensity=propensity_vector,
            outcome_mean=None if outcome_mean is None else as_vector(outcome_mean, "outcome_mean"),
        )
        if outcome_mean_vector is None:
            raise ValueError("Overlap-targeted diagnostics require `outcome_mean` or enough nuisance information to infer it.")
        overlap_pseudo_outcome, overlap_weights = _r_pseudo_target(
            treatment=treatment_vector,
            outcome=outcome_vector,
            outcome_mean=outcome_mean_vector,
            propensity=propensity_vector,
        )
        effective_weight = [
            base_weight * overlap_weight
            for base_weight, overlap_weight in zip(sample_weight_vector, overlap_weights)
        ]
        validate_nonnegative_weights(effective_weight, "overlap_effective_weight")
        overlap_result = _build_target_result(
            predictions=prediction_vector,
            pseudo_outcome=overlap_pseudo_outcome,
            sample_weight=effective_weight,
            method=method_name,
            method_options=method_options,
            fold_ids=fold_vector,
            level=confidence_level,
            target_population="overlap",
            comparison_predictions=comparison_vector,
        )

    primary = overlap_result if target_population == "overlap" else dr_result
    if primary is None:
        raise RuntimeError("Failed to construct the requested diagnostics target.")  # pragma: no cover
    return CalibrationDiagnostics(
        estimate=primary.estimate,
        plugin_estimate=primary.plugin_estimate,
        standard_error=primary.standard_error,
        confidence_interval=primary.confidence_interval,
        curve_predictions=primary.curve_predictions,
        curve_estimates=primary.curve_estimates,
        method=primary.method,
        n_folds=primary.n_folds,
        target_population=target_population,
        overlap_diagnostics=overlap,
        fold_estimates=primary.fold_estimates,
        comparison_estimate=primary.comparison_estimate,
        comparison_standard_error=primary.comparison_standard_error,
        dr_result=dr_result,
        overlap_result=overlap_result,
    )
