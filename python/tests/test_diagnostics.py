"""Diagnostics tests for the Python API."""

from __future__ import annotations

import unittest
import csv
import math
import os

from causal_calibration import diagnose_calibration


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_returns_interval_and_fold_estimates(self) -> None:
        diagnostics = diagnose_calibration(
            predictions=[0.1, 0.2, 0.4, 0.5, 0.8],
            treatment=[0.0, 1.0, 0.0, 1.0, 1.0],
            outcome=[0.2, 0.4, 0.5, 0.8, 1.1],
            mu0=[0.1, 0.2, 0.3, 0.4, 0.5],
            mu1=[0.3, 0.5, 0.6, 0.9, 1.2],
            propensity=[0.4, 0.6, 0.5, 0.5, 0.7],
            curve_method="histogram",
            jackknife_folds=5,
        )
        self.assertEqual(len(diagnostics.fold_estimates), 5)
        self.assertLessEqual(diagnostics.confidence_interval[0], diagnostics.confidence_interval[1])
        self.assertGreaterEqual(diagnostics.standard_error, 0.0)

    def test_overlap_target_diagnostics_returns_both_targets(self) -> None:
        diagnostics = diagnose_calibration(
            predictions=[0.1, 0.2, 0.4, 0.5, 0.8],
            treatment=[0.0, 1.0, 0.0, 1.0, 1.0],
            outcome=[0.2, 0.4, 0.5, 0.8, 1.1],
            mu0=[0.1, 0.2, 0.3, 0.4, 0.5],
            mu1=[0.3, 0.5, 0.6, 0.9, 1.2],
            propensity=[0.4, 0.6, 0.5, 0.5, 0.7],
            outcome_mean=[0.18, 0.32, 0.45, 0.6, 0.95],
            curve_method="histogram",
            target_population="both",
            jackknife_folds=5,
        )
        self.assertIsNotNone(diagnostics.dr_result)
        self.assertIsNotNone(diagnostics.overlap_result)
        self.assertEqual(diagnostics.target_population, "both")
        self.assertIsNotNone(diagnostics.overlap_diagnostics)

    def test_shared_fixture_diagnostics_match(self) -> None:
        fixture_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "shared", "fixtures")
        )
        with open(os.path.join(fixture_dir, "core_fixture.csv"), newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with open(os.path.join(fixture_dir, "expected_outputs.csv"), newline="", encoding="utf-8") as handle:
            expected = list(csv.DictReader(handle))
        diagnostics = diagnose_calibration(
            predictions=[float(row["prediction"]) for row in rows],
            treatment=[float(row["treatment"]) for row in rows],
            outcome=[float(row["outcome"]) for row in rows],
            mu0=[float(row["mu0"]) for row in rows],
            mu1=[float(row["mu1"]) for row in rows],
            propensity=[float(row["propensity"]) for row in rows],
            curve_method="histogram",
        )
        want_estimate = next(
            float(row["value"])
            for row in expected
            if row["kind"] == "diagnostics" and row["metric"] == "estimate"
        )
        want_se = next(
            float(row["value"])
            for row in expected
            if row["kind"] == "diagnostics" and row["metric"] == "standard_error"
        )
        self.assertTrue(math.isclose(diagnostics.estimate, want_estimate, rel_tol=1e-8, abs_tol=1e-8))
        self.assertTrue(math.isclose(diagnostics.standard_error, want_se, rel_tol=1e-8, abs_tol=1e-8))

    def test_plotting_helpers_return_axes(self) -> None:
        diagnostics = diagnose_calibration(
            predictions=[0.1, 0.2, 0.4, 0.5, 0.8],
            treatment=[0.0, 1.0, 0.0, 1.0, 1.0],
            outcome=[0.2, 0.4, 0.5, 0.8, 1.1],
            mu0=[0.1, 0.2, 0.3, 0.4, 0.5],
            mu1=[0.3, 0.5, 0.6, 0.9, 1.2],
            propensity=[0.4, 0.6, 0.5, 0.5, 0.7],
            curve_method="histogram",
            jackknife_folds=5,
        )
        self.assertIsNotNone(diagnostics.plot())
        self.assertIsNotNone(diagnostics.overlap_diagnostics.plot())


if __name__ == "__main__":
    unittest.main()
