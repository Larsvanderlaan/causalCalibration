#' Construct a calibration bundle from vectors
#'
#' @param predictions Numeric vector of treatment-effect predictions.
#' @param treatment Numeric binary vector of treatment assignments.
#' @param outcome Numeric vector of observed outcomes.
#' @param mu0 Optional control outcome regression estimates.
#' @param mu1 Optional treated outcome regression estimates.
#' @param outcome_mean Optional marginal outcome estimates.
#' @param propensity Optional propensity estimates.
#' @param sample_weight Optional observation weights.
#'
#' @return An object of class `"calibration_bundle"`.
#' @export
calibration_bundle <- function(
  predictions,
  treatment,
  outcome,
  mu0 = NULL,
  mu1 = NULL,
  outcome_mean = NULL,
  propensity = NULL,
  sample_weight = NULL
) {
  structure(
    list(
      predictions = .cc_as_numeric_vector(predictions, "predictions"),
      treatment = .cc_as_numeric_vector(treatment, "treatment"),
      outcome = .cc_as_numeric_vector(outcome, "outcome"),
      mu0 = if (is.null(mu0)) NULL else .cc_as_numeric_vector(mu0, "mu0"),
      mu1 = if (is.null(mu1)) NULL else .cc_as_numeric_vector(mu1, "mu1"),
      outcome_mean = if (is.null(outcome_mean)) NULL else .cc_as_numeric_vector(outcome_mean, "outcome_mean"),
      propensity = if (is.null(propensity)) NULL else .cc_as_numeric_vector(propensity, "propensity"),
      sample_weight = if (is.null(sample_weight)) NULL else .cc_as_numeric_vector(sample_weight, "sample_weight")
    ),
    class = "calibration_bundle"
  )
}

#' Construct a calibration bundle from a data frame
#'
#' @param data A `data.frame`.
#' @param predictions,treatment,outcome,mu0,mu1,outcome_mean,propensity,sample_weight Column names.
#'
#' @return An object of class `"calibration_bundle"`.
#' @export
calibration_bundle_from_data_frame <- function(
  data,
  predictions = "predictions",
  treatment = "treatment",
  outcome = "outcome",
  mu0 = "mu0",
  mu1 = "mu1",
  outcome_mean = "outcome_mean",
  propensity = "propensity",
  sample_weight = "sample_weight"
) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame.", call. = FALSE)
  }
  calibration_bundle(
    predictions = data[[predictions]],
    treatment = data[[treatment]],
    outcome = data[[outcome]],
    mu0 = if (is.null(mu0)) NULL else data[[mu0]],
    mu1 = if (is.null(mu1)) NULL else data[[mu1]],
    outcome_mean = if (is.null(outcome_mean)) NULL else data[[outcome_mean]],
    propensity = if (is.null(propensity)) NULL else data[[propensity]],
    sample_weight = if (is.null(sample_weight)) NULL else data[[sample_weight]]
  )
}

#' Construct a cross-fit bundle from vectors
#'
#' @inheritParams calibration_bundle
#' @param fold_predictions Numeric matrix of fold-specific predictions.
#' @param fold_ids Optional integer vector identifying the out-of-fold column for each observation.
#'
#' @return An object of class `"crossfit_bundle"`.
#' @export
crossfit_bundle <- function(
  predictions,
  fold_predictions,
  treatment,
  outcome,
  mu0 = NULL,
  mu1 = NULL,
  outcome_mean = NULL,
  propensity = NULL,
  sample_weight = NULL,
  fold_ids = NULL
) {
  base <- calibration_bundle(
    predictions = predictions,
    treatment = treatment,
    outcome = outcome,
    mu0 = mu0,
    mu1 = mu1,
    outcome_mean = outcome_mean,
    propensity = propensity,
    sample_weight = sample_weight
  )
  structure(
    c(
      base,
      list(
        fold_predictions = .cc_as_matrix(fold_predictions, "fold_predictions"),
        fold_ids = if (is.null(fold_ids)) NULL else as.integer(fold_ids)
      )
    ),
    class = c("crossfit_bundle", "calibration_bundle")
  )
}

#' Construct a cross-fit bundle from a data frame
#'
#' @param data A `data.frame`.
#' @param fold_prediction_columns Character vector naming fold-specific prediction columns.
#' @param fold_ids Optional fold-id column name.
#' @inheritParams calibration_bundle_from_data_frame
#'
#' @return An object of class `"crossfit_bundle"`.
#' @export
crossfit_bundle_from_data_frame <- function(
  data,
  fold_prediction_columns,
  fold_ids = NULL,
  predictions = "predictions",
  treatment = "treatment",
  outcome = "outcome",
  mu0 = "mu0",
  mu1 = "mu1",
  outcome_mean = "outcome_mean",
  propensity = "propensity",
  sample_weight = "sample_weight"
) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame.", call. = FALSE)
  }
  crossfit_bundle(
    predictions = data[[predictions]],
    fold_predictions = as.matrix(data[, fold_prediction_columns, drop = FALSE]),
    treatment = data[[treatment]],
    outcome = data[[outcome]],
    mu0 = if (is.null(mu0)) NULL else data[[mu0]],
    mu1 = if (is.null(mu1)) NULL else data[[mu1]],
    outcome_mean = if (is.null(outcome_mean)) NULL else data[[outcome_mean]],
    propensity = if (is.null(propensity)) NULL else data[[propensity]],
    sample_weight = if (is.null(sample_weight)) NULL else data[[sample_weight]],
    fold_ids = if (is.null(fold_ids)) NULL else data[[fold_ids]]
  )
}

#' Validate a bundle object
#'
#' @param bundle A calibration or cross-fit bundle.
#' @param tolerance Alignment tolerance for cross-fit validation.
#'
#' @return A named list summarizing the validated bundle.
#' @export
validate_bundle <- function(bundle, tolerance = 1e-8) {
  if (inherits(bundle, "crossfit_bundle")) {
    return(
      validate_crossfit_bundle(
        predictions = bundle$predictions,
        fold_predictions = bundle$fold_predictions,
        fold_ids = bundle$fold_ids,
        tolerance = tolerance
      )
    )
  }
  if (!inherits(bundle, "calibration_bundle")) {
    stop("`bundle` must inherit from `calibration_bundle` or `crossfit_bundle`.", call. = FALSE)
  }
  .cc_validate_same_length(
    length(bundle$predictions),
    treatment = bundle$treatment,
    outcome = bundle$outcome
  )
  if (!is.null(bundle$mu0)) {
    .cc_validate_same_length(length(bundle$predictions), mu0 = bundle$mu0)
  }
  if (!is.null(bundle$mu1)) {
    .cc_validate_same_length(length(bundle$predictions), mu1 = bundle$mu1)
  }
  if (!is.null(bundle$outcome_mean)) {
    .cc_validate_same_length(length(bundle$predictions), outcome_mean = bundle$outcome_mean)
  }
  if (!is.null(bundle$propensity)) {
    .cc_validate_same_length(length(bundle$predictions), propensity = bundle$propensity)
  }
  if (!is.null(bundle$sample_weight)) {
    .cc_validate_same_length(length(bundle$predictions), sample_weight = bundle$sample_weight)
  }
  list(n_obs = length(bundle$predictions))
}

#' Fit a calibrator from a bundle
#'
#' @param bundle A `calibration_bundle`.
#' @param ... Additional arguments passed to [fit_calibrator()].
#'
#' @return A fitted calibrator.
#' @export
fit_bundle_calibrator <- function(bundle, ...) {
  if (!inherits(bundle, "calibration_bundle")) {
    stop("`bundle` must inherit from `calibration_bundle`.", call. = FALSE)
  }
  fit_calibrator(
    predictions = bundle$predictions,
    treatment = bundle$treatment,
    outcome = bundle$outcome,
    mu0 = bundle$mu0,
    mu1 = bundle$mu1,
    outcome_mean = bundle$outcome_mean,
    propensity = bundle$propensity,
    sample_weight = bundle$sample_weight,
    ...
  )
}

#' Fit a cross-calibrator from a cross-fit bundle
#'
#' @param bundle A `crossfit_bundle`.
#' @param ... Additional arguments passed to [fit_cross_calibrator()].
#'
#' @return A fitted cross-calibrator.
#' @export
fit_bundle_cross_calibrator <- function(bundle, ...) {
  if (!inherits(bundle, "crossfit_bundle")) {
    stop("`bundle` must inherit from `crossfit_bundle`.", call. = FALSE)
  }
  fit_cross_calibrator(
    predictions = bundle$predictions,
    fold_predictions = bundle$fold_predictions,
    fold_ids = bundle$fold_ids,
    treatment = bundle$treatment,
    outcome = bundle$outcome,
    mu0 = bundle$mu0,
    mu1 = bundle$mu1,
    outcome_mean = bundle$outcome_mean,
    propensity = bundle$propensity,
    sample_weight = bundle$sample_weight,
    ...
  )
}

#' @export
print.calibration_bundle <- function(x, ...) {
  cat("<calibration_bundle>\n")
  cat(sprintf("  n_obs: %s\n", length(x$predictions)))
  invisible(x)
}

#' @export
print.crossfit_bundle <- function(x, ...) {
  cat("<crossfit_bundle>\n")
  cat(sprintf("  n_obs: %s\n", length(x$predictions)))
  cat(sprintf("  n_folds: %s\n", ncol(x$fold_predictions)))
  invisible(x)
}
