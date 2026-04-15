fixture_path <- function(name) {
  testthat::test_path("..", "..", "..", "..", "shared", "fixtures", name)
}

read_fixture <- function(name) {
  utils::read.csv(fixture_path(name), stringsAsFactors = FALSE)
}

test_that("linear calibrator keeps a nonnegative slope", {
  calibrator <- fit_calibrator(
    predictions = c(0, 1, 2),
    treatment = c(0, 1, 0),
    outcome = c(3, 2, 1),
    loss = "dr",
    method = "linear",
    mu0 = c(1, 1, 1),
    mu1 = c(0, 0, 0),
    propensity = c(0.5, 0.5, 0.5)
  )
  expect_gte(calibrator$model$slope, 0)
})

test_that("shared fixture expected outputs match", {
  rows <- read_fixture("core_fixture.csv")
  expected <- read_fixture("expected_outputs.csv")
  prediction_grid <- rows$grid_prediction
  for (loss in c("dr", "r")) {
    for (method in c("isotonic", "smooth_isotonic", "linear", "histogram")) {
      calibrator <- fit_calibrator(
        predictions = rows$prediction,
        treatment = rows$treatment,
        outcome = rows$outcome,
        loss = loss,
        method = method,
        mu0 = rows$mu0,
        mu1 = rows$mu1,
        outcome_mean = rows$outcome_mean,
        propensity = rows$propensity
      )
      got <- predict(calibrator, prediction_grid)
      want <- expected$value[expected$kind == "calibration" & expected$loss == loss & expected$method == method]
      expect_equal(got, want, tolerance = 1e-8)

      fold_mat <- as.matrix(rows[, c("fold_1", "fold_2", "fold_3")])
      cross_calibrator <- fit_cross_calibrator(
        predictions = rows$prediction,
        fold_predictions = fold_mat,
        treatment = rows$treatment,
        outcome = rows$outcome,
        loss = loss,
        method = method,
        mu0 = rows$mu0,
        mu1 = rows$mu1,
        outcome_mean = rows$outcome_mean,
        propensity = rows$propensity
      )
      got_cross <- predict(cross_calibrator, fold_mat)
      want_cross <- expected$value[expected$kind == "cross" & expected$loss == loss & expected$method == method]
      expect_equal(got_cross, want_cross, tolerance = 1e-8)
    }
  }
})

test_that("shared fixture diagnostics match", {
  rows <- read_fixture("core_fixture.csv")
  expected <- read_fixture("expected_outputs.csv")
  diagnostics <- diagnose_calibration(
    predictions = rows$prediction,
    treatment = rows$treatment,
    outcome = rows$outcome,
    mu0 = rows$mu0,
    mu1 = rows$mu1,
    propensity = rows$propensity,
    curve_method = "histogram"
  )
  want_estimate <- expected$value[expected$kind == "diagnostics" & expected$metric == "estimate"]
  want_se <- expected$value[expected$kind == "diagnostics" & expected$metric == "standard_error"]
  expect_equal(diagnostics$estimate, want_estimate, tolerance = 1e-8)
  expect_equal(diagnostics$standard_error, want_se, tolerance = 1e-8)
})
