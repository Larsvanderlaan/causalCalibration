"""Core calibration APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from warnings import warn

from ._algorithms import (
    HistogramModel,
    LightGBMIsotonicModel,
    LinearModel,
    MonotoneSplineModel,
    fit_backend,
)
from ._utils import (
    as_matrix_rows,
    as_optional_vector,
    as_vector,
    clip_propensity,
    mapping_grid,
    order_statistic_median,
    validate_binary,
    validate_fold_ids,
    validate_method,
    validate_min_unique_scores,
    validate_nonnegative_weights,
    validate_oof_alignment,
    validate_same_length,
)
from .overlap import OverlapDiagnostics, assess_overlap

CalibrationModel = HistogramModel | LightGBMIsotonicModel | MonotoneSplineModel | LinearModel


def _dr_pseudo_outcome(
    treatment: list[float],
    outcome: list[float],
    mu0: list[float],
    mu1: list[float],
    propensity: list[float],
) -> list[float]:
    pseudo_outcome: list[float] = []
    for a_value, y_value, mu0_value, mu1_value, propensity_value in zip(
        treatment, outcome, mu0, mu1, propensity
    ):
        mu_observed = mu1_value if a_value == 1.0 else mu0_value
        propensity_observed = propensity_value if a_value == 1.0 else 1.0 - propensity_value
        pseudo_outcome.append(
            mu1_value
            - mu0_value
            + ((2.0 * a_value - 1.0) / propensity_observed) * (y_value - mu_observed)
        )
    return pseudo_outcome


def _r_pseudo_target(
    treatment: list[float],
    outcome: list[float],
    outcome_mean: list[float],
    propensity: list[float],
) -> tuple[list[float], list[float]]:
    pseudo_outcome: list[float] = []
    pseudo_weight: list[float] = []
    for a_value, y_value, m_value, propensity_value in zip(
        treatment, outcome, outcome_mean, propensity
    ):
        residual = a_value - propensity_value
        if residual == 0:
            raise ValueError("R-loss residualized treatment is zero; adjust propensity clipping.")
        pseudo_outcome.append((y_value - m_value) / residual)
        pseudo_weight.append(residual * residual)
    return pseudo_outcome, pseudo_weight


def _emit_overlap_warnings(overlap: OverlapDiagnostics, loss: str) -> None:
    for message in overlap.recommendation_messages():
        if overlap.recommended_loss != loss or overlap.severity == "severe":
            warn(message, stacklevel=3)


@dataclass
class Calibrator:
    """Fitted calibrator."""

    loss: str
    method: str
    model: CalibrationModel
    clip: float
    n_obs: int
    method_options: dict[str, float] = field(default_factory=dict)
    training_predictions: list[float] = field(default_factory=list)
    fitted_values: list[float] = field(default_factory=list)
    effective_weights: list[float] = field(default_factory=list)
    overlap_diagnostics: OverlapDiagnostics | None = None

    def predict(self, predictions: Any) -> list[float] | float:
        vector = as_vector(predictions, "predictions")
        output = self.model.predict_many(vector)
        if isinstance(predictions, (int, float)):
            return output[0]
        return output

    __call__ = predict

    def summary(self) -> dict[str, Any]:
        summary = {
            "loss": self.loss,
            "method": self.method,
            "n_obs": self.n_obs,
            "clip": self.clip,
            "method_options": dict(self.method_options),
        }
        if self.overlap_diagnostics is not None:
            summary["overlap"] = self.overlap_diagnostics.summary()
        return summary

    def mapping_frame(self, n_points: int = 200) -> list[dict[str, float]]:
        if isinstance(self.model, HistogramModel):
            return self.model.mapping_rows()
        grid = mapping_grid(self.training_predictions, n_points=n_points)
        values = self.model.predict_many(grid)
        return [{"prediction": point, "calibrated": value} for point, value in zip(grid, values)]

    def plot(self, ax: Any | None = None, *, n_points: int = 200) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Plotting calibrators requires matplotlib.") from exc
        if ax is None:
            _, ax = plt.subplots()
        rows = self.mapping_frame(n_points=n_points)
        if rows and "lower" in rows[0]:
            x_values = [row["lower"] for row in rows]
            y_values = [row["value"] for row in rows]
            ax.step(x_values, y_values, where="post")
        else:
            ax.plot([row["prediction"] for row in rows], [row["calibrated"] for row in rows])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Calibrated prediction")
        ax.set_title(f"{self.method} calibration")
        return ax


@dataclass
class CrossCalibrator:
    """Cross-calibrator using fold-specific predictions."""

    calibrator: Calibrator
    aggregation: str = "median"
    n_folds: int | None = None
    fold_ids: list[int] | None = None
    validation_tolerance: float = 1e-8

    def predict(self, fold_predictions: Any) -> list[float] | float:
        if isinstance(fold_predictions, (int, float)):
            return self.calibrator.predict(fold_predictions)
        matrix = as_matrix_rows(fold_predictions, "fold_predictions")
        if self.n_folds is not None and any(len(row) != self.n_folds for row in matrix):
            raise ValueError(f"Each row of `fold_predictions` must have {self.n_folds} columns.")
        calibrated_rows = [self.calibrator.model.predict_many(row) for row in matrix]
        output = [order_statistic_median(row) for row in calibrated_rows]
        if matrix and len(matrix[0]) == 1:
            return output
        return output

    __call__ = predict

    def summary(self) -> dict[str, Any]:
        summary = self.calibrator.summary()
        summary.update({"aggregation": self.aggregation, "n_folds": self.n_folds})
        if self.fold_ids is not None:
            summary["has_fold_ids"] = True
        return summary

    def plot(self, ax: Any | None = None, *, n_points: int = 200) -> Any:
        return self.calibrator.plot(ax=ax, n_points=n_points)


def validate_crossfit_bundle(
    *,
    predictions: list[float] | tuple[float, ...],
    fold_predictions: list[list[float]] | tuple[tuple[float, ...], ...],
    fold_ids: list[int] | tuple[int, ...] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    prediction_vector = as_vector(predictions, "predictions")
    matrix = as_matrix_rows(fold_predictions, "fold_predictions")
    if len(matrix) != len(prediction_vector):
        raise ValueError("`fold_predictions` must have one row per observation in `predictions`.")
    summary: dict[str, float] = {
        "n_obs": float(len(prediction_vector)),
        "n_folds": float(len(matrix[0])),
    }
    if fold_ids is not None:
        fold_vector = validate_fold_ids(len(prediction_vector), fold_ids)
        if max(fold_vector) != len(matrix[0]):
            raise ValueError("`fold_ids` must align with the number of columns in `fold_predictions`.")
        summary.update(validate_oof_alignment(prediction_vector, matrix, fold_vector, tolerance=tolerance))
        summary["has_fold_ids"] = 1.0
    else:
        summary["has_fold_ids"] = 0.0
    return summary


def fit_calibrator(
    *,
    predictions: list[float] | tuple[float, ...],
    treatment: list[float] | tuple[float, ...],
    outcome: list[float] | tuple[float, ...],
    loss: str = "dr",
    method: str = "isotonic",
    mu0: list[float] | tuple[float, ...] | None = None,
    mu1: list[float] | tuple[float, ...] | None = None,
    outcome_mean: list[float] | tuple[float, ...] | None = None,
    propensity: list[float] | tuple[float, ...] | None = None,
    sample_weight: list[float] | tuple[float, ...] | None = None,
    clip: float = 1e-6,
    method_options: dict[str, float] | None = None,
) -> Calibrator:
    prediction_vector = as_vector(predictions, "predictions")
    n_obs = len(prediction_vector)
    treatment_vector = as_vector(treatment, "treatment")
    outcome_vector = as_vector(outcome, "outcome")
    validate_same_length(n_obs, treatment=treatment_vector, outcome=outcome_vector)
    validate_binary(treatment_vector, "treatment")
    sample_weight_vector = as_optional_vector(sample_weight, "sample_weight", n_obs)
    validate_nonnegative_weights(sample_weight_vector)
    method_name = validate_method(method)
    validate_min_unique_scores(prediction_vector, method_name)

    propensity_raw = as_vector(propensity, "propensity") if propensity is not None else None
    propensity_vector = clip_propensity(propensity_raw, clip) if propensity_raw is not None else None
    overlap = None
    if propensity_raw is not None:
        validate_same_length(n_obs, propensity=propensity_raw)
        overlap = assess_overlap(
            treatment=treatment_vector,
            propensity=propensity_raw,
            sample_weight=sample_weight_vector,
            clip=clip,
        )
        _emit_overlap_warnings(overlap, loss=loss)

    if loss == "dr":
        if mu0 is None or mu1 is None or propensity_vector is None:
            raise ValueError("DR loss requires `mu0`, `mu1`, and `propensity`.")
        mu0_vector = as_vector(mu0, "mu0")
        mu1_vector = as_vector(mu1, "mu1")
        validate_same_length(n_obs, mu0=mu0_vector, mu1=mu1_vector, propensity=propensity_vector)
        pseudo_outcome = _dr_pseudo_outcome(
            treatment=treatment_vector,
            outcome=outcome_vector,
            mu0=mu0_vector,
            mu1=mu1_vector,
            propensity=propensity_vector,
        )
        effective_weight = sample_weight_vector
    elif loss == "r":
        if outcome_mean is None or propensity_vector is None:
            raise ValueError("R loss requires `outcome_mean` and `propensity`.")
        outcome_mean_vector = as_vector(outcome_mean, "outcome_mean")
        validate_same_length(n_obs, outcome_mean=outcome_mean_vector, propensity=propensity_vector)
        pseudo_outcome, r_weight = _r_pseudo_target(
            treatment=treatment_vector,
            outcome=outcome_vector,
            outcome_mean=outcome_mean_vector,
            propensity=propensity_vector,
        )
        effective_weight = [
            base_weight * residual_weight
            for base_weight, residual_weight in zip(sample_weight_vector, r_weight)
        ]
        validate_nonnegative_weights(effective_weight, "effective_weight")
    else:
        raise ValueError("`loss` must be either 'dr' or 'r'.")

    model = fit_backend(
        method=method_name,
        x=prediction_vector,
        y=pseudo_outcome,
        weights=effective_weight,
        method_options=method_options,
    )
    fitted_values = model.predict_many(prediction_vector)
    return Calibrator(
        loss=loss,
        method=method_name,
        model=model,
        clip=clip,
        n_obs=n_obs,
        method_options=method_options or {},
        training_predictions=prediction_vector,
        fitted_values=fitted_values,
        effective_weights=effective_weight,
        overlap_diagnostics=overlap,
    )


def fit_cross_calibrator(
    *,
    predictions: list[float] | tuple[float, ...],
    fold_predictions: list[list[float]] | tuple[tuple[float, ...], ...],
    treatment: list[float] | tuple[float, ...],
    outcome: list[float] | tuple[float, ...],
    loss: str = "dr",
    method: str = "isotonic",
    mu0: list[float] | tuple[float, ...] | None = None,
    mu1: list[float] | tuple[float, ...] | None = None,
    outcome_mean: list[float] | tuple[float, ...] | None = None,
    propensity: list[float] | tuple[float, ...] | None = None,
    sample_weight: list[float] | tuple[float, ...] | None = None,
    clip: float = 1e-6,
    method_options: dict[str, float] | None = None,
    fold_ids: list[int] | tuple[int, ...] | None = None,
    validation_tolerance: float = 1e-8,
) -> CrossCalibrator:
    matrix = as_matrix_rows(fold_predictions, "fold_predictions")
    validate_crossfit_bundle(
        predictions=predictions,
        fold_predictions=matrix,
        fold_ids=fold_ids,
        tolerance=validation_tolerance,
    )
    calibrator = fit_calibrator(
        predictions=predictions,
        treatment=treatment,
        outcome=outcome,
        loss=loss,
        method=method,
        mu0=mu0,
        mu1=mu1,
        outcome_mean=outcome_mean,
        propensity=propensity,
        sample_weight=sample_weight,
        clip=clip,
        method_options=method_options,
    )
    fold_vector = None if fold_ids is None else validate_fold_ids(len(matrix), fold_ids)
    return CrossCalibrator(
        calibrator=calibrator,
        n_folds=len(matrix[0]),
        fold_ids=fold_vector,
        validation_tolerance=validation_tolerance,
    )
