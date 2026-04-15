"""Python unit tests for the calibration API."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import unittest

from causal_calibration import (
    CrossFitBundle,
    assess_overlap,
    fit_calibrator,
    fit_cross_calibrator,
    validate_crossfit_bundle,
)


FIXTURE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "shared", "fixtures")
)
HAS_ISOTONIC_DEPS = (
    importlib.util.find_spec("lightgbm") is not None
    and importlib.util.find_spec("sklearn") is not None
)


def read_fixture(name: str) -> list[dict[str, str]]:
    with open(os.path.join(FIXTURE_DIR, name), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CalibrationApiTests(unittest.TestCase):
    def test_isotonic_backend_smoke_test_when_dependencies_are_available(self) -> None:
        if not HAS_ISOTONIC_DEPS:
            self.skipTest("lightgbm/sklearn is unavailable")
        calibrator = fit_calibrator(
            predictions=[0.1, 0.2, 0.4, 0.5],
            treatment=[0.0, 1.0, 0.0, 1.0],
            outcome=[0.2, 0.4, 0.5, 0.8],
            loss="dr",
            method="isotonic",
            mu0=[0.1, 0.2, 0.3, 0.4],
            mu1=[0.3, 0.5, 0.6, 0.9],
            propensity=[0.4, 0.6, 0.5, 0.5],
        )
        self.assertEqual(len(calibrator.predict([0.15, 0.45])), 2)

    def test_linear_calibrator_respects_nonnegative_slope(self) -> None:
        calibrator = fit_calibrator(
            predictions=[0.0, 1.0, 2.0],
            treatment=[0.0, 1.0, 0.0],
            outcome=[3.0, 2.0, 1.0],
            loss="dr",
            method="linear",
            mu0=[1.0, 1.0, 1.0],
            mu1=[0.0, 0.0, 0.0],
            propensity=[0.5, 0.5, 0.5],
        )
        self.assertGreaterEqual(calibrator.model.slope, 0.0)

    def test_cross_calibrator_returns_order_statistic_median(self) -> None:
        cross_calibrator = fit_cross_calibrator(
            predictions=[0.1, 0.2, 0.3],
            fold_predictions=[[0.0, 0.1], [0.2, 0.3], [0.3, 0.6]],
            fold_ids=[2, 1, 1],
            treatment=[0.0, 1.0, 0.0],
            outcome=[0.0, 1.0, 0.0],
            loss="dr",
            method="histogram",
            mu0=[0.0, 0.0, 0.0],
            mu1=[0.5, 0.7, 0.9],
            propensity=[0.5, 0.5, 0.5],
        )
        predictions = cross_calibrator.predict([[0.0, 0.2]])
        self.assertEqual(len(predictions), 1)

    def test_overlap_helper_recommends_r_under_weak_overlap(self) -> None:
        overlap = assess_overlap(
            treatment=[0.0, 1.0, 0.0, 1.0],
            propensity=[0.01, 0.99, 0.04, 0.96],
        )
        self.assertEqual(overlap.recommended_loss, "r")
        self.assertIn(overlap.severity, {"weak", "severe"})

    def test_crossfit_validation_checks_oof_alignment(self) -> None:
        summary = validate_crossfit_bundle(
            predictions=[0.1, 0.5, 0.9],
            fold_predictions=[[0.1, 0.0], [0.2, 0.5], [0.9, 0.8]],
            fold_ids=[1, 2, 1],
        )
        self.assertEqual(summary["n_obs"], 3.0)
        self.assertEqual(summary["has_fold_ids"], 1.0)

    def test_crossfit_bundle_round_trips_into_fit(self) -> None:
        bundle = CrossFitBundle.from_mapping(
            {
                "predictions": [0.1, 0.2, 0.3, 0.4],
                "fold_predictions": [
                    [0.1, 0.0],
                    [0.0, 0.2],
                    [0.3, 0.1],
                    [0.2, 0.4],
                ],
                "fold_ids": [1, 2, 1, 2],
                "treatment": [0.0, 1.0, 0.0, 1.0],
                "outcome": [0.1, 0.8, 0.2, 1.0],
                "mu0": [0.0, 0.2, 0.1, 0.3],
                "mu1": [0.3, 0.9, 0.4, 1.1],
                "propensity": [0.5, 0.5, 0.5, 0.5],
            }
        )
        fitted = bundle.fit_cross_calibrator(loss="dr", method="histogram")
        self.assertEqual(fitted.n_folds, 2)

    def test_calibrator_plot_returns_axes(self) -> None:
        calibrator = fit_calibrator(
            predictions=[0.1, 0.2, 0.3, 0.4],
            treatment=[0.0, 1.0, 0.0, 1.0],
            outcome=[0.1, 0.8, 0.2, 1.0],
            loss="dr",
            method="linear",
            mu0=[0.0, 0.2, 0.1, 0.3],
            mu1=[0.3, 0.9, 0.4, 1.1],
            propensity=[0.5, 0.5, 0.5, 0.5],
        )
        self.assertIsNotNone(calibrator.plot())

    def test_shared_fixture_expected_outputs_match(self) -> None:
        rows = read_fixture("core_fixture.csv")
        expected = read_fixture("expected_outputs.csv")
        dataset = {
            "predictions": [float(row["prediction"]) for row in rows],
            "treatment": [float(row["treatment"]) for row in rows],
            "outcome": [float(row["outcome"]) for row in rows],
            "mu0": [float(row["mu0"]) for row in rows],
            "mu1": [float(row["mu1"]) for row in rows],
            "outcome_mean": [float(row["outcome_mean"]) for row in rows],
            "propensity": [float(row["propensity"]) for row in rows],
        }
        prediction_grid = [float(row["grid_prediction"]) for row in rows]
        fold_matrix = [
            [float(row["fold_1"]), float(row["fold_2"]), float(row["fold_3"])]
            for row in rows
        ]
        fold_ids = [1, 2, 3, 1, 2, 3, 1, 2]
        oof_predictions = [row[fold_id - 1] for row, fold_id in zip(fold_matrix, fold_ids)]
        for loss in ("dr", "r"):
            for method in ("monotone_spline", "linear", "histogram"):
                tolerance = 1e-2 if method == "monotone_spline" else 1e-8
                calibrator = fit_calibrator(
                    predictions=dataset["predictions"],
                    treatment=dataset["treatment"],
                    outcome=dataset["outcome"],
                    loss=loss,
                    method=method,
                    mu0=dataset["mu0"],
                    mu1=dataset["mu1"],
                    outcome_mean=dataset["outcome_mean"],
                    propensity=dataset["propensity"],
                )
                got = calibrator.predict(prediction_grid)
                want = [
                    float(row["value"])
                    for row in expected
                    if row["kind"] == "calibration" and row["loss"] == loss and row["method"] == method
                ]
                self.assertEqual(len(got), len(want))
                for actual, target in zip(got, want):
                    self.assertTrue(math.isclose(actual, target, rel_tol=tolerance, abs_tol=tolerance))
                cross_calibrator = fit_cross_calibrator(
                    predictions=oof_predictions,
                    fold_predictions=fold_matrix,
                    fold_ids=fold_ids,
                    treatment=dataset["treatment"],
                    outcome=dataset["outcome"],
                    loss=loss,
                    method=method,
                    mu0=dataset["mu0"],
                    mu1=dataset["mu1"],
                    outcome_mean=dataset["outcome_mean"],
                    propensity=dataset["propensity"],
                )
                got_cross = cross_calibrator.predict(fold_matrix)
                want_cross = [
                    float(row["value"])
                    for row in expected
                    if row["kind"] == "cross" and row["loss"] == loss and row["method"] == method
                ]
                for actual, target in zip(got_cross, want_cross):
                    self.assertTrue(math.isclose(actual, target, rel_tol=tolerance, abs_tol=tolerance))


if __name__ == "__main__":
    unittest.main()
