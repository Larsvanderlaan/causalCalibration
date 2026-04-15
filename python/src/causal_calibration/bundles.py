"""Workflow bundle helpers for external prediction pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._utils import (
    as_matrix_rows,
    as_optional_vector,
    as_vector,
    get_column,
    validate_same_length,
)


def _frame_column(frame: Any, name: str) -> Any:
    if isinstance(frame, Mapping):
        return frame[name]
    return frame[name]


@dataclass
class CalibrationBundle:
    predictions: list[float]
    treatment: list[float]
    outcome: list[float]
    mu0: list[float] | None = None
    mu1: list[float] | None = None
    outcome_mean: list[float] | None = None
    propensity: list[float] | None = None
    sample_weight: list[float] | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CalibrationBundle":
        return cls(
            predictions=as_vector(get_column(mapping, "predictions"), "predictions"),
            treatment=as_vector(get_column(mapping, "treatment"), "treatment"),
            outcome=as_vector(get_column(mapping, "outcome"), "outcome"),
            mu0=None if get_column(mapping, "mu0", required=False) is None else as_vector(get_column(mapping, "mu0"), "mu0"),
            mu1=None if get_column(mapping, "mu1", required=False) is None else as_vector(get_column(mapping, "mu1"), "mu1"),
            outcome_mean=None
            if get_column(mapping, "outcome_mean", required=False) is None
            else as_vector(get_column(mapping, "outcome_mean"), "outcome_mean"),
            propensity=None
            if get_column(mapping, "propensity", required=False) is None
            else as_vector(get_column(mapping, "propensity"), "propensity"),
            sample_weight=None
            if get_column(mapping, "sample_weight", required=False) is None
            else as_vector(get_column(mapping, "sample_weight"), "sample_weight"),
        )

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        predictions: str = "predictions",
        treatment: str = "treatment",
        outcome: str = "outcome",
        mu0: str | None = "mu0",
        mu1: str | None = "mu1",
        outcome_mean: str | None = "outcome_mean",
        propensity: str | None = "propensity",
        sample_weight: str | None = "sample_weight",
    ) -> "CalibrationBundle":
        return cls(
            predictions=as_vector(_frame_column(frame, predictions), "predictions"),
            treatment=as_vector(_frame_column(frame, treatment), "treatment"),
            outcome=as_vector(_frame_column(frame, outcome), "outcome"),
            mu0=None if mu0 is None else as_vector(_frame_column(frame, mu0), "mu0"),
            mu1=None if mu1 is None else as_vector(_frame_column(frame, mu1), "mu1"),
            outcome_mean=None if outcome_mean is None else as_vector(_frame_column(frame, outcome_mean), "outcome_mean"),
            propensity=None if propensity is None else as_vector(_frame_column(frame, propensity), "propensity"),
            sample_weight=None if sample_weight is None else as_vector(_frame_column(frame, sample_weight), "sample_weight"),
        )

    def validate(self) -> dict[str, int]:
        validate_same_length(
            len(self.predictions),
            treatment=self.treatment,
            outcome=self.outcome,
        )
        if self.mu0 is not None:
            validate_same_length(len(self.predictions), mu0=self.mu0)
        if self.mu1 is not None:
            validate_same_length(len(self.predictions), mu1=self.mu1)
        if self.outcome_mean is not None:
            validate_same_length(len(self.predictions), outcome_mean=self.outcome_mean)
        if self.propensity is not None:
            validate_same_length(len(self.predictions), propensity=self.propensity)
        if self.sample_weight is not None:
            validate_same_length(len(self.predictions), sample_weight=self.sample_weight)
        return {"n_obs": len(self.predictions)}

    def fit_calibrator(self, **kwargs: Any) -> Any:
        from .core import fit_calibrator

        payload: dict[str, Any] = {
            "predictions": self.predictions,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "mu0": self.mu0,
            "mu1": self.mu1,
            "outcome_mean": self.outcome_mean,
            "propensity": self.propensity,
            "sample_weight": self.sample_weight,
        }
        payload.update(kwargs)
        return fit_calibrator(**payload)


@dataclass
class CrossFitBundle(CalibrationBundle):
    fold_predictions: list[list[float]] | None = None
    fold_ids: list[int] | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CrossFitBundle":
        base = CalibrationBundle.from_mapping(mapping)
        return cls(
            predictions=base.predictions,
            treatment=base.treatment,
            outcome=base.outcome,
            mu0=base.mu0,
            mu1=base.mu1,
            outcome_mean=base.outcome_mean,
            propensity=base.propensity,
            sample_weight=base.sample_weight,
            fold_predictions=as_matrix_rows(get_column(mapping, "fold_predictions"), "fold_predictions"),
            fold_ids=None
            if get_column(mapping, "fold_ids", required=False) is None
            else [int(value) for value in as_vector(get_column(mapping, "fold_ids"), "fold_ids")],
        )

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        fold_prediction_columns: list[str],
        fold_ids: str | None = None,
        **kwargs: Any,
    ) -> "CrossFitBundle":
        base = CalibrationBundle.from_frame(frame, **kwargs)
        fold_matrix = as_matrix_rows(
            [[_frame_column(frame, column)[index] for column in fold_prediction_columns] for index in range(len(base.predictions))],
            "fold_predictions",
        )
        fold_id_vector = None
        if fold_ids is not None:
            fold_id_vector = [int(value) for value in as_vector(_frame_column(frame, fold_ids), "fold_ids")]
        return cls(
            predictions=base.predictions,
            treatment=base.treatment,
            outcome=base.outcome,
            mu0=base.mu0,
            mu1=base.mu1,
            outcome_mean=base.outcome_mean,
            propensity=base.propensity,
            sample_weight=base.sample_weight,
            fold_predictions=fold_matrix,
            fold_ids=fold_id_vector,
        )

    def validate(self, tolerance: float = 1e-8) -> dict[str, float]:
        from .core import validate_crossfit_bundle

        if self.fold_predictions is None:
            raise ValueError("`fold_predictions` must be supplied for a CrossFitBundle.")
        return validate_crossfit_bundle(
            predictions=self.predictions,
            fold_predictions=self.fold_predictions,
            fold_ids=self.fold_ids,
            tolerance=tolerance,
        )

    def fit_cross_calibrator(self, **kwargs: Any) -> Any:
        from .core import fit_cross_calibrator

        if self.fold_predictions is None:
            raise ValueError("`fold_predictions` must be supplied for a CrossFitBundle.")
        payload: dict[str, Any] = {
            "predictions": self.predictions,
            "fold_predictions": self.fold_predictions,
            "fold_ids": self.fold_ids,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "mu0": self.mu0,
            "mu1": self.mu1,
            "outcome_mean": self.outcome_mean,
            "propensity": self.propensity,
            "sample_weight": self.sample_weight,
        }
        payload.update(kwargs)
        return fit_cross_calibrator(**payload)
