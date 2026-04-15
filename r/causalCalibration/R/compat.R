#' Deprecated compatibility wrapper for the paper-era API
#'
#' @inheritParams fit_calibrator
#' @param tau Deprecated alias for `predictions`.
#' @param A Deprecated alias for `treatment`.
#' @param Y Deprecated alias for `outcome`.
#' @param EY1 Deprecated alias for `mu1`.
#' @param EY0 Deprecated alias for `mu0`.
#' @param pA1 Deprecated alias for `propensity`.
#' @param tau_pred Optional prediction vector to calibrate immediately.
#'
#' @return A legacy-style list compatible with the original package.
#' @export
causalCalibrate <- function(tau, A, Y, EY1, EY0, pA1, weights = NULL, tau_pred = tau) {
  .Deprecated("fit_calibrator")
  calibrator <- fit_calibrator(
    predictions = tau,
    treatment = A,
    outcome = Y,
    loss = "dr",
    method = "isotonic",
    mu0 = EY0,
    mu1 = EY1,
    propensity = pA1,
    sample_weight = weights
  )
  calibration_function <- function(new_tau) {
    predict(calibrator, new_tau)
  }
  list(
    tau_calibrated = calibration_function(tau_pred),
    calibration_function = calibration_function,
    calibrator = calibrator
  )
}

#' Deprecated compatibility wrapper for paper-era cross-calibration
#'
#' @param output Output from [causalCalibrate()] or [fit_calibrator()].
#' @param new_tau_mat Matrix of fold-specific predictions.
#'
#' @return Numeric vector of cross-calibrated predictions.
#' @export
cross_calibrate <- function(output, new_tau_mat) {
  .Deprecated("fit_cross_calibrator")
  calibrator <- if (inherits(output, "causal_calibrator")) {
    output
  } else if (!is.null(output$calibrator)) {
    output$calibrator
  } else {
    stop("`output` must come from `causalCalibrate()` or `fit_calibrator()`.", call. = FALSE)
  }
  matrix_data <- .cc_as_matrix(new_tau_mat, "new_tau_mat")
  calibrated_matrix <- apply(matrix_data, 2, function(column) predict(calibrator, column))
  if (is.vector(calibrated_matrix)) {
    calibrated_matrix <- matrix(calibrated_matrix, ncol = ncol(matrix_data))
  }
  apply(calibrated_matrix, 1, .cc_order_statistic_median)
}
