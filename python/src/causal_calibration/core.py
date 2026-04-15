"""Core calibration APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._algorithms import LinearModel, SmoothModel, StepModel, fit_backend
from ._utils import (
    as_matrix_rows,
    as_optional_vector,
    as_vector,
    clip_propensity,
    mapping_grid,
    order_statistic_median,
    validate_binary,
    validate_same_length,
)


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
        pseudo_outcome.append(mu1_value - mu0_value + ((2.0 * a_value - 1.0) / propensity_observed) * (y_value - mu_observed))
    return pseudo_outcome


def _r_pseudo_target(
    treatment: list[float],
    outcome: list[float],
    outcome_mean: list[float],
    propensity: list[float],
) -> tuple[list[float], list[float]]:
    pseudo_outcome: list[float] = []
    pseudo_weight: list[float] = []
    for a_value, y_value, m_value, propensity_value in zip(treatment, outcome, outcome_mean, propensity):
        residual = a_value - propensity_value
        if residual == 0:
            raise ValueError("R-loss residualized treatment is zero; adjust propensity clipping.")
        pseudo_outcome.append((y_value - m_value) / residual)
        pseudo_weight.append(residual * residual)
    return pseudo_outcome, pseudo_weight


@dataclass
class Calibrator:
    """Fitted calibrator."""

    loss: str
    method: str
    model: StepModel | SmoothModel | LinearModel
    clip: float
    n_obs: int
    method_options: dict[str, float] = field(default_factory=dict)
    training_predictions: list[float] = field(default_factory=list)
    fitted_values: list[float] = field(default_factory=list)
    effective_weights: list[float] = field(default_factory=list)

    def predict(self, predictions: Any) -> list[float] | float:
        vector = as_vector(predictions, "predictions")
        output = self.model.predict_many(vector)
        if isinstance(predictions, (int, float)):
            return output[0]
        return output

    __call__ = predict

    def summary(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "method": self.method,
            "n_obs": self.n_obs,
            "clip": self.clip,
            "method_options": dict(self.method_options),
        }

    def mapping_frame(self, n_points: int = 200) -> list[dict[str, float]]:
        if self.method in {"isotonic", "histogram"}:
            return self.model.mapping_rows()
        grid = mapping_grid(self.training_predictions, n_points=n_points)
        values = self.model.predict_many(grid)
        return [{"prediction": point, "calibrated": value} for point, value in zip(grid, values)]


@dataclass
class CrossCalibrator:
    """Cross-calibrator using fold-specific predictions."""

    calibrator: Calibrator
    aggregation: str = "median"
    n_folds: int | None = None

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
    propensity_vector = clip_propensity(as_vector(propensity, "propensity"), clip) if propensity is not None else None

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
        effective_weight = [base_weight * residual_weight for base_weight, residual_weight in zip(sample_weight_vector, r_weight)]
    else:
        raise ValueError("`loss` must be either 'dr' or 'r'.")

    model = fit_backend(
        method=method,
        x=prediction_vector,
        y=pseudo_outcome,
        weights=effective_weight,
        method_options=method_options,
    )
    fitted_values = model.predict_many(prediction_vector)
    return Calibrator(
        loss=loss,
        method=method,
        model=model,
        clip=clip,
        n_obs=n_obs,
        method_options=method_options or {},
        training_predictions=prediction_vector,
        fitted_values=fitted_values,
        effective_weights=effective_weight,
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
) -> CrossCalibrator:
    matrix = as_matrix_rows(fold_predictions, "fold_predictions")
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
    return CrossCalibrator(calibrator=calibrator, n_folds=len(matrix[0]))
