"""Minimal Python example."""

from causal_calibration import diagnose_calibration, fit_calibrator


predictions = [0.2, 0.8, 0.1, 0.6]
treatment = [0, 1, 0, 1]
outcome = [0.1, 0.9, 0.0, 0.7]
mu0 = [0.05, 0.5, 0.02, 0.4]
mu1 = [0.3, 0.95, 0.2, 0.8]
propensity = [0.3, 0.7, 0.25, 0.6]

calibrator = fit_calibrator(
    predictions=predictions,
    treatment=treatment,
    outcome=outcome,
    mu0=mu0,
    mu1=mu1,
    propensity=propensity,
    loss="dr",
    method="isotonic",
)

print(calibrator.predict(predictions))

diagnostics = diagnose_calibration(
    predictions=predictions,
    treatment=treatment,
    outcome=outcome,
    mu0=mu0,
    mu1=mu1,
    propensity=propensity,
)

print(diagnostics.summary())
