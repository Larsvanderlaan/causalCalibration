"""Python unit tests for the calibration API."""

from __future__ import annotations

import csv
import math
import os
import unittest

from causal_calibration import fit_calibrator, fit_cross_calibrator


FIXTURE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "shared", "fixtures")
)


def read_fixture(name: str) -> list[dict[str, str]]:
    with open(os.path.join(FIXTURE_DIR, name), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CalibrationApiTests(unittest.TestCase):
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
            fold_predictions=[[0.0, 0.1, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.6]],
            treatment=[0.0, 1.0, 0.0],
            outcome=[0.0, 1.0, 0.0],
            loss="dr",
            method="histogram",
            mu0=[0.0, 0.0, 0.0],
            mu1=[0.5, 0.7, 0.9],
            propensity=[0.5, 0.5, 0.5],
        )
        predictions = cross_calibrator.predict([[0.0, 0.2, 0.8]])
        self.assertEqual(len(predictions), 1)

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
        for loss in ("dr", "r"):
            for method in ("isotonic", "smooth_isotonic", "linear", "histogram"):
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
                    self.assertTrue(math.isclose(actual, target, rel_tol=1e-8, abs_tol=1e-8))
                cross_calibrator = fit_cross_calibrator(
                    predictions=dataset["predictions"],
                    fold_predictions=[
                        [float(row["fold_1"]), float(row["fold_2"]), float(row["fold_3"])]
                        for row in rows
                    ],
                    treatment=dataset["treatment"],
                    outcome=dataset["outcome"],
                    loss=loss,
                    method=method,
                    mu0=dataset["mu0"],
                    mu1=dataset["mu1"],
                    outcome_mean=dataset["outcome_mean"],
                    propensity=dataset["propensity"],
                )
                got_cross = cross_calibrator.predict(
                    [
                        [float(row["fold_1"]), float(row["fold_2"]), float(row["fold_3"])]
                        for row in rows
                    ]
                )
                want_cross = [
                    float(row["value"])
                    for row in expected
                    if row["kind"] == "cross" and row["loss"] == loss and row["method"] == method
                ]
                for actual, target in zip(got_cross, want_cross):
                    self.assertTrue(math.isclose(actual, target, rel_tol=1e-8, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
