#' Fit a cross-calibrator
#'
#' Fits a calibrator on pooled out-of-fold predictions and stores the expected
#' number of fold-specific prediction columns for cross-calibrated prediction.
#'
#' @param predictions Numeric vector of pooled out-of-fold predictions.
#' @param fold_predictions Numeric matrix with one column per fold-specific
#'   predictor.
#' @inheritParams fit_calibrator
#'
#' @return A fitted object of class `"causal_cross_calibrator"`.
#' @export
fit_cross_calibrator <- function(
  predictions,
  fold_predictions,
  treatment,
  outcome,
  loss = "dr",
  method = "isotonic",
  mu0 = NULL,
  mu1 = NULL,
  outcome_mean = NULL,
  propensity = NULL,
  sample_weight = NULL,
  clip = 1e-6,
  method_options = list()
) {
  fold_predictions <- .cc_as_matrix(fold_predictions, "fold_predictions")
  calibrator <- fit_calibrator(
    predictions = predictions,
    treatment = treatment,
    outcome = outcome,
    loss = loss,
    method = method,
    mu0 = mu0,
    mu1 = mu1,
    outcome_mean = outcome_mean,
    propensity = propensity,
    sample_weight = sample_weight,
    clip = clip,
    method_options = method_options
  )
  structure(
    list(
      calibrator = calibrator,
      aggregation = "median",
      n_folds = ncol(fold_predictions)
    ),
    class = "causal_cross_calibrator"
  )
}

#' @export
predict.causal_cross_calibrator <- function(object, newdata, ...) {
  if (is.atomic(newdata) && !is.matrix(newdata)) {
    return(predict(object$calibrator, newdata))
  }
  matrix_data <- .cc_as_matrix(newdata, "newdata")
  if (ncol(matrix_data) != object$n_folds) {
    stop(sprintf("`newdata` must have %s columns.", object$n_folds), call. = FALSE)
  }
  calibrated_matrix <- apply(matrix_data, 2, function(column) .cc_predict_backend(object$calibrator$model, column))
  if (is.vector(calibrated_matrix)) {
    calibrated_matrix <- matrix(calibrated_matrix, ncol = object$n_folds)
  }
  apply(calibrated_matrix, 1, .cc_order_statistic_median)
}

#' @export
summary.causal_cross_calibrator <- function(object, ...) {
  summary(object$calibrator)
}

#' @export
print.causal_cross_calibrator <- function(x, ...) {
  cat("<causal_cross_calibrator>\n")
  cat(sprintf("  method: %s\n", x$calibrator$method))
  cat(sprintf("  loss: %s\n", x$calibrator$loss))
  cat(sprintf("  n_folds: %s\n", x$n_folds))
  invisible(x)
}
