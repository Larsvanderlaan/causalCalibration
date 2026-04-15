.cc_linear_loo_curve <- function(predictions, pseudo_outcome, sample_weight) {
  sum_w <- sum(sample_weight)
  sum_x <- sum(sample_weight * predictions)
  sum_y <- sum(sample_weight * pseudo_outcome)
  sum_xx <- sum(sample_weight * predictions^2)
  sum_xy <- sum(sample_weight * predictions * pseudo_outcome)
  vapply(
    seq_along(predictions),
    function(index) {
      remaining_w <- sum_w - sample_weight[index]
      if (remaining_w <= 0) {
        return(pseudo_outcome[index])
      }
      remaining_x <- sum_x - sample_weight[index] * predictions[index]
      remaining_y <- sum_y - sample_weight[index] * pseudo_outcome[index]
      remaining_xx <- sum_xx - sample_weight[index] * predictions[index]^2
      remaining_xy <- sum_xy - sample_weight[index] * predictions[index] * pseudo_outcome[index]
      mean_x <- remaining_x / remaining_w
      mean_y <- remaining_y / remaining_w
      denominator <- remaining_xx - remaining_w * mean_x^2
      slope <- if (denominator == 0) {
        0
      } else {
        max(0, (remaining_xy - remaining_w * mean_x * mean_y) / denominator)
      }
      intercept <- mean_y - slope * mean_x
      intercept + slope * predictions[index]
    },
    numeric(1)
  )
}

.cc_histogram_loo_curve <- function(predictions, pseudo_outcome, sample_weight, method_options) {
  n_bins <- if (!is.null(method_options$n_bins)) as.integer(method_options$n_bins) else 10L
  histogram <- .cc_fit_histogram(predictions, pseudo_outcome, sample_weight, n_bins = n_bins)
  bin_ids <- findInterval(predictions, histogram$lower_bounds)
  bin_ids[bin_ids < 1L] <- 1L
  bin_weight <- rowsum(sample_weight, group = bin_ids, reorder = FALSE)
  bin_sum <- rowsum(sample_weight * pseudo_outcome, group = bin_ids, reorder = FALSE)
  full_weight <- sum(sample_weight)
  full_sum <- sum(sample_weight * pseudo_outcome)
  vapply(
    seq_along(predictions),
    function(index) {
      bin_index <- bin_ids[index]
      remaining_weight <- bin_weight[bin_index, 1] - sample_weight[index]
      if (remaining_weight > 0) {
        return((bin_sum[bin_index, 1] - sample_weight[index] * pseudo_outcome[index]) / remaining_weight)
      }
      if (bin_index > 1L) {
        return(histogram$values[bin_index - 1L])
      }
      (full_sum - sample_weight[index] * pseudo_outcome[index]) / max(full_weight - sample_weight[index], 1e-12)
    },
    numeric(1)
  )
}

.cc_estimate_curve <- function(predictions, pseudo_outcome, sample_weight, method, method_options, fold_ids) {
  model <- .cc_fit_backend(method, predictions, pseudo_outcome, sample_weight, method_options = method_options)
  fitted <- .cc_predict_backend(model, predictions)
  if (method == "linear") {
    oof_curve <- .cc_linear_loo_curve(predictions, pseudo_outcome, sample_weight)
  } else if (method == "histogram") {
    oof_curve <- .cc_histogram_loo_curve(predictions, pseudo_outcome, sample_weight, method_options)
  } else {
    oof_curve <- numeric(length(predictions))
    unique_folds <- sort(unique(fold_ids))
    for (fold in unique_folds) {
      keep <- which(fold_ids != fold)
      holdout <- which(fold_ids == fold)
      oof_model <- .cc_fit_backend(
        method,
        predictions[keep],
        pseudo_outcome[keep],
        sample_weight[keep],
        method_options = method_options
      )
      oof_curve[holdout] <- .cc_predict_backend(oof_model, predictions[holdout])
    }
  }
  plugin_terms <- (fitted - predictions)^2
  robust_terms <- (oof_curve - predictions)^2 + 2 * (oof_curve - predictions) * (pseudo_outcome - oof_curve)
  list(
    model = model,
    fitted = fitted,
    oof_curve = oof_curve,
    plugin_estimate = .cc_weighted_mean(plugin_terms, sample_weight),
    estimate = .cc_weighted_mean(robust_terms, sample_weight)
  )
}

.cc_diagnose_one <- function(
  predictions,
  treatment,
  outcome,
  mu0,
  mu1,
  propensity,
  sample_weight,
  curve_method,
  method_options,
  fold_ids,
  confidence_level
) {
  pseudo_outcome <- .cc_dr_pseudo_outcome(treatment, outcome, mu0, mu1, propensity)
  full <- .cc_estimate_curve(predictions, pseudo_outcome, sample_weight, curve_method, method_options, fold_ids)
  unique_folds <- sort(unique(fold_ids))
  fold_estimates <- vapply(
    unique_folds,
    function(fold) {
      keep <- which(fold_ids != fold)
      subset <- .cc_estimate_curve(
        predictions[keep],
        pseudo_outcome[keep],
        sample_weight[keep],
        curve_method,
        method_options,
        fold_ids[keep]
      )
      subset$estimate
    },
    numeric(1)
  )
  mean_estimate <- mean(fold_estimates)
  variance <- ((length(unique_folds) - 1) / length(unique_folds)) * sum((fold_estimates - mean_estimate)^2)
  standard_error <- sqrt(variance)
  z_score <- .cc_normal_quantile(confidence_level)
  grid <- .cc_mapping_grid(predictions)
  grid_estimates <- .cc_predict_backend(full$model, grid)
  structure(
    list(
      estimate = full$estimate,
      plugin_estimate = full$plugin_estimate,
      standard_error = standard_error,
      confidence_interval = c(full$estimate - z_score * standard_error, full$estimate + z_score * standard_error),
      curve_predictions = grid,
      curve_estimates = grid_estimates,
      curve_method = curve_method,
      jackknife_folds = length(unique_folds),
      fold_estimates = fold_estimates,
      comparison_estimate = NULL,
      comparison_standard_error = NULL
    ),
    class = "causal_calibration_diagnostics"
  )
}

#' Diagnose calibration error with a doubly robust estimator
#'
#' Estimates a calibration curve and L2 calibration error using the doubly robust
#' calibration-error estimator described by Xu and Yadlowsky (2022), together
#' with deterministic K-fold jackknife standard errors.
#'
#' @param predictions Numeric vector of treatment-effect predictions to evaluate.
#' @param treatment Numeric binary vector of treatment assignments.
#' @param outcome Numeric vector of observed outcomes.
#' @param mu0 Numeric vector of estimated control outcome regressions.
#' @param mu1 Numeric vector of estimated treated outcome regressions.
#' @param propensity Numeric vector of estimated propensity scores.
#' @param sample_weight Optional numeric vector of observation weights.
#' @param comparison_predictions Optional second prediction vector used for
#'   before/after comparisons.
#' @param curve_method Calibration curve backend used for diagnostics.
#' @param method_options Optional named list of backend-specific settings.
#' @param fold_ids Optional integer vector of jackknife fold assignments.
#' @param jackknife_folds Number of deterministic balanced folds when `fold_ids`
#'   is not supplied.
#' @param clip Propensity clipping threshold.
#' @param confidence_level Confidence level for the normal-approximation interval.
#'
#' @return A diagnostics object of class `"causal_calibration_diagnostics"`.
#' @export
diagnose_calibration <- function(
  predictions,
  treatment,
  outcome,
  mu0,
  mu1,
  propensity,
  sample_weight = NULL,
  comparison_predictions = NULL,
  curve_method = "histogram",
  method_options = list(),
  fold_ids = NULL,
  jackknife_folds = 100L,
  clip = 1e-6,
  confidence_level = 0.95
) {
  predictions <- .cc_as_numeric_vector(predictions, "predictions")
  treatment <- .cc_as_numeric_vector(treatment, "treatment")
  outcome <- .cc_as_numeric_vector(outcome, "outcome")
  mu0 <- .cc_as_numeric_vector(mu0, "mu0")
  mu1 <- .cc_as_numeric_vector(mu1, "mu1")
  propensity <- .cc_clip_propensity(.cc_as_numeric_vector(propensity, "propensity"), clip)
  sample_weight <- if (is.null(sample_weight)) rep(1, length(predictions)) else .cc_as_numeric_vector(sample_weight, "sample_weight")
  .cc_validate_same_length(
    length(predictions),
    treatment = treatment,
    outcome = outcome,
    mu0 = mu0,
    mu1 = mu1,
    propensity = propensity,
    sample_weight = sample_weight
  )
  .cc_validate_binary(treatment, "treatment")
  fold_ids <- if (is.null(fold_ids)) .cc_balanced_folds(length(predictions), jackknife_folds) else as.integer(fold_ids)
  .cc_validate_same_length(length(predictions), fold_ids = fold_ids)

  diagnostics <- .cc_diagnose_one(
    predictions = predictions,
    treatment = treatment,
    outcome = outcome,
    mu0 = mu0,
    mu1 = mu1,
    propensity = propensity,
    sample_weight = sample_weight,
    curve_method = curve_method,
    method_options = method_options,
    fold_ids = fold_ids,
    confidence_level = confidence_level
  )
  if (!is.null(comparison_predictions)) {
    comparison_predictions <- .cc_as_numeric_vector(comparison_predictions, "comparison_predictions")
    .cc_validate_same_length(length(predictions), comparison_predictions = comparison_predictions)
    comparison <- .cc_diagnose_one(
      predictions = comparison_predictions,
      treatment = treatment,
      outcome = outcome,
      mu0 = mu0,
      mu1 = mu1,
      propensity = propensity,
      sample_weight = sample_weight,
      curve_method = curve_method,
      method_options = method_options,
      fold_ids = fold_ids,
      confidence_level = confidence_level
    )
    diagnostics$comparison_estimate <- comparison$estimate
    diagnostics$comparison_standard_error <- comparison$standard_error
  }
  diagnostics
}

#' @export
summary.causal_calibration_diagnostics <- function(object, ...) {
  summary <- list(
    estimate = object$estimate,
    plugin_estimate = object$plugin_estimate,
    standard_error = object$standard_error,
    confidence_interval = object$confidence_interval,
    curve_method = object$curve_method,
    jackknife_folds = object$jackknife_folds
  )
  if (!is.null(object$comparison_estimate)) {
    summary$comparison_estimate <- object$comparison_estimate
    summary$comparison_standard_error <- object$comparison_standard_error
    summary$improvement <- object$comparison_estimate - object$estimate
  }
  summary
}

#' @export
print.causal_calibration_diagnostics <- function(x, ...) {
  cat("<causal_calibration_diagnostics>\n")
  cat(sprintf("  estimate: %.6f\n", x$estimate))
  cat(sprintf("  standard_error: %.6f\n", x$standard_error))
  cat(sprintf("  interval: [%.6f, %.6f]\n", x$confidence_interval[1], x$confidence_interval[2]))
  invisible(x)
}

#' @export
plot.causal_calibration_diagnostics <- function(x, ...) {
  plot(
    x$curve_predictions,
    x$curve_estimates,
    type = "l",
    xlab = "Prediction",
    ylab = "Estimated calibration curve",
    ...
  )
}
