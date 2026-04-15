predictions <- c(0.2, 0.8, 0.1, 0.6)
treatment <- c(0, 1, 0, 1)
outcome <- c(0.1, 0.9, 0.0, 0.7)
mu0 <- c(0.05, 0.5, 0.02, 0.4)
mu1 <- c(0.3, 0.95, 0.2, 0.8)
propensity <- c(0.3, 0.7, 0.25, 0.6)

calibrator <- fit_calibrator(
  predictions = predictions,
  treatment = treatment,
  outcome = outcome,
  mu0 = mu0,
  mu1 = mu1,
  propensity = propensity,
  loss = "dr",
  method = "isotonic"
)

print(predict(calibrator, predictions))
print(summary(assess_overlap(treatment = treatment, propensity = propensity)))

diagnostics <- diagnose_calibration(
  predictions = predictions,
  treatment = treatment,
  outcome = outcome,
  mu0 = mu0,
  mu1 = mu1,
  propensity = propensity,
  target_population = "both"
)

print(summary(diagnostics))
