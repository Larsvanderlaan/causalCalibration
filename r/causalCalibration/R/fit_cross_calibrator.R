#' Validate a cross-fit prediction bundle
#'
#' @param predictions Numeric vector of pooled out-of-fold predictions.
#' @param fold_predictions Numeric matrix with one column per fold-specific predictor.
#' @param fold_ids Optional integer vector identifying the out-of-fold column for each observation.
#' @param tolerance Absolute tolerance for pooled-vs-fold out-of-fold alignment checks.
#'
#' @return A named list describing the validated bundle.
#' @export
validate_crossfit_bundle <- function(
  predictions,
  fold_predictions,
  fold_ids = NULL,
  tolerance = 1e-8
) {
  predictions <- .cc_as_numeric_vector(predictions, "predictions")
  fold_predictions <- .cc_as_matrix(fold_predictions, "fold_predictions")
  if (nrow(fold_predictions) != length(predictions)) {
    stop("`fold_predictions` must have one row per observation in `predictions`.", call. = FALSE)
  }
  summary <- list(
    n_obs = length(predictions),
    n_folds = ncol(fold_predictions),
    has_fold_ids = FALSE
  )
  if (!is.null(fold_ids)) {
    fold_ids <- .cc_validate_fold_ids(fold_ids, length(predictions))
    if (max(fold_ids) != ncol(fold_predictions)) {
      stop("`fold_ids` must align with the number of columns in `fold_predictions`.", call. = FALSE)
    }
    summary$has_fold_ids <- TRUE
    summary <- c(summary, .cc_validate_oof_alignment(predictions, fold_predictions, fold_ids, tolerance = tolerance))
  }
  summary
}

#' Fit a cross-calibrator
#'
#' Fits a calibrator on pooled out-of-fold predictions and stores the expected
#' number of fold-specific prediction columns for cross-calibrated prediction.
#'
#' @param predictions Numeric vector of pooled out-of-fold predictions.
#' @param fold_predictions Numeric matrix with one column per fold-specific
#'   predictor.
#' @param fold_ids Optional integer vector identifying the out-of-fold column
#'   for each observation.
#' @param validation_tolerance Absolute tolerance for pooled-vs-fold alignment
#'   checks when `fold_ids` is supplied.
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
  method_options = list(),
  fold_ids = NULL,
  validation_tolerance = 1e-8
) {
  fold_predictions <- .cc_as_matrix(fold_predictions, "fold_predictions")
  validate_crossfit_bundle(
    predictions = predictions,
    fold_predictions = fold_predictions,
    fold_ids = fold_ids,
    tolerance = validation_tolerance
  )
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
      n_folds = ncol(fold_predictions),
      fold_ids = if (is.null(fold_ids)) NULL else .cc_validate_fold_ids(fold_ids, nrow(fold_predictions)),
      validation_tolerance = validation_tolerance
    ),
    class = "causal_cross_calibrator"
  )
}

#' @export
predict.causal_cross_calibrator <- function(object, newdata, ...) {
  if (is.atomic(newdata) && !is.matrix(newdata) && !is.data.frame(newdata)) {
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
  summary <- summary(object$calibrator)
  summary$aggregation <- object$aggregation
  summary$n_folds <- object$n_folds
  summary$has_fold_ids <- !is.null(object$fold_ids)
  summary
}

#' @export
print.causal_cross_calibrator <- function(x, ...) {
  cat("<causal_cross_calibrator>\n")
  cat(sprintf("  method: %s\n", x$calibrator$method))
  cat(sprintf("  loss: %s\n", x$calibrator$loss))
  cat(sprintf("  n_folds: %s\n", x$n_folds))
  invisible(x)
}
