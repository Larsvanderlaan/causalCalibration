.cc_linear_loo_curve <- function(predictions, pseudo_outcome, sample_weight) {
  sum_w <- sum(sample_weight)
  sum_x <- sum(sample_weight * predictions)
  sum_y <- sum(sample_weight * pseudo_outcome)
  sum_xx <- sum(sample_weight * predictions^2)
  sum_xy <- sum(sample_weight * predictions * pseudo_outcome)
  vapply(
    seq_along(predictions),
    function(index) {
      remaining_w <- sum_w - sample_weight[[index]]
      if (remaining_w <= 0) {
        return(pseudo_outcome[[index]])
      }
      remaining_x <- sum_x - sample_weight[[index]] * predictions[[index]]
      remaining_y <- sum_y - sample_weight[[index]] * pseudo_outcome[[index]]
      remaining_xx <- sum_xx - sample_weight[[index]] * predictions[[index]]^2
      remaining_xy <- sum_xy - sample_weight[[index]] * predictions[[index]] * pseudo_outcome[[index]]
      mean_x <- remaining_x / remaining_w
      mean_y <- remaining_y / remaining_w
      denominator <- remaining_xx - remaining_w * mean_x^2
      slope <- if (denominator == 0) {
        0
      } else {
        max(0, (remaining_xy - remaining_w * mean_x * mean_y) / denominator)
      }
      intercept <- mean_y - slope * mean_x
      intercept + slope * predictions[[index]]
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
      bin_index <- bin_ids[[index]]
      remaining_weight <- bin_weight[bin_index, 1] - sample_weight[[index]]
      if (remaining_weight > 0) {
        return((bin_sum[bin_index, 1] - sample_weight[[index]] * pseudo_outcome[[index]]) / remaining_weight)
      }
      if (bin_index > 1L) {
        return(histogram$values[[bin_index - 1L]])
      }
      (full_sum - sample_weight[[index]] * pseudo_outcome[[index]]) / max(full_weight - sample_weight[[index]], 1e-12)
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
  robust_terms <- (pseudo_outcome - predictions) * (oof_curve - predictions)
  list(
    model = model,
    fitted = fitted,
    oof_curve = oof_curve,
    plugin_estimate = .cc_weighted_mean(plugin_terms, sample_weight),
    estimate = .cc_weighted_mean(robust_terms, sample_weight)
  )
}

.cc_build_target_result <- function(
  predictions,
  pseudo_outcome,
  sample_weight,
  method,
  method_options,
  fold_ids,
  confidence_level,
  target_population,
  comparison_predictions = NULL
) {
  full <- .cc_estimate_curve(predictions, pseudo_outcome, sample_weight, method, method_options, fold_ids)
  unique_folds <- sort(unique(fold_ids))
  fold_estimates <- vapply(
    unique_folds,
    function(fold) {
      keep <- which(fold_ids != fold)
      subset <- .cc_estimate_curve(
        predictions[keep],
        pseudo_outcome[keep],
        sample_weight[keep],
        method,
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
  result <- list(
    target_population = target_population,
    estimate = full$estimate,
    plugin_estimate = full$plugin_estimate,
    standard_error = standard_error,
    confidence_interval = c(full$estimate - z_score * standard_error, full$estimate + z_score * standard_error),
    curve_predictions = grid,
    curve_estimates = grid_estimates,
    curve_method = method,
    jackknife_folds = length(unique_folds),
    fold_estimates = fold_estimates,
    comparison_estimate = NULL,
    comparison_standard_error = NULL
  )
  if (!is.null(comparison_predictions)) {
    comparison <- .cc_estimate_curve(
      comparison_predictions,
      pseudo_outcome,
      sample_weight,
      method,
      method_options,
      fold_ids
    )
    comparison_fold_estimates <- vapply(
      unique_folds,
      function(fold) {
        keep <- which(fold_ids != fold)
        subset <- .cc_estimate_curve(
          comparison_predictions[keep],
          pseudo_outcome[keep],
          sample_weight[keep],
          method,
          method_options,
          fold_ids[keep]
        )
        subset$estimate
      },
      numeric(1)
    )
    mean_comparison <- mean(comparison_fold_estimates)
    variance_comparison <- ((length(unique_folds) - 1) / length(unique_folds)) * sum((comparison_fold_estimates - mean_comparison)^2)
    result$comparison_estimate <- comparison$estimate
    result$comparison_standard_error <- sqrt(variance_comparison)
  }
  class(result) <- "causal_calibration_target_result"
  result
}

#' Diagnose calibration error
#'
#' Estimates a calibration curve and L2 calibration error using doubly robust
#' and overlap-weighted calibration targets with deterministic K-fold jackknife
#' standard errors.
#'
#' @param predictions Numeric vector of treatment-effect predictions to evaluate.
#' @param treatment Numeric binary vector of treatment assignments.
#' @param outcome Numeric vector of observed outcomes.
#' @param mu0 Numeric vector of estimated control outcome regressions.
#' @param mu1 Numeric vector of estimated treated outcome regressions.
#' @param propensity Numeric vector of estimated propensity scores.
#' @param outcome_mean Optional marginal outcome estimates for overlap-targeted diagnostics.
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
#' @param target_population One of `"dr"`, `"overlap"`, or `"both"`.
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
  outcome_mean = NULL,
  sample_weight = NULL,
  comparison_predictions = NULL,
  curve_method = "histogram",
  method_options = list(),
  fold_ids = NULL,
  jackknife_folds = 100L,
  clip = 1e-6,
  confidence_level = 0.95,
  target_population = "dr"
) {
  if (!(target_population %in% c("dr", "overlap", "both"))) {
    stop("`target_population` must be one of 'dr', 'overlap', or 'both'.", call. = FALSE)
  }
  predictions <- .cc_as_numeric_vector(predictions, "predictions")
  treatment <- .cc_as_numeric_vector(treatment, "treatment")
  outcome <- .cc_as_numeric_vector(outcome, "outcome")
  mu0 <- .cc_as_numeric_vector(mu0, "mu0")
  mu1 <- .cc_as_numeric_vector(mu1, "mu1")
  propensity_raw <- .cc_as_numeric_vector(propensity, "propensity")
  propensity <- .cc_clip_propensity(propensity_raw, clip)
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
  .cc_validate_nonnegative_weights(sample_weight)
  curve_method <- .cc_validate_method(curve_method)
  .cc_validate_min_unique_scores(predictions, curve_method)
  fold_ids <- if (is.null(fold_ids)) .cc_balanced_folds(length(predictions), jackknife_folds) else .cc_validate_fold_ids(fold_ids, length(predictions))

  overlap <- assess_overlap(
    treatment = treatment,
    propensity = propensity_raw,
    sample_weight = sample_weight,
    clip = clip
  )
  for (message in .cc_overlap_messages(overlap)) {
    warning(message, call. = FALSE, immediate. = TRUE)
  }

  if (!is.null(comparison_predictions)) {
    comparison_predictions <- .cc_as_numeric_vector(comparison_predictions, "comparison_predictions")
    .cc_validate_same_length(length(predictions), comparison_predictions = comparison_predictions)
  }

  dr_result <- NULL
  overlap_result <- NULL
  if (target_population %in% c("dr", "both")) {
    dr_result <- .cc_build_target_result(
      predictions = predictions,
      pseudo_outcome = .cc_dr_pseudo_outcome(treatment, outcome, mu0, mu1, propensity),
      sample_weight = sample_weight,
      method = curve_method,
      method_options = method_options,
      fold_ids = fold_ids,
      confidence_level = confidence_level,
      target_population = "dr",
      comparison_predictions = comparison_predictions
    )
  }

  if (target_population %in% c("overlap", "both")) {
    outcome_mean <- .cc_infer_outcome_mean(
      mu0 = mu0,
      mu1 = mu1,
      propensity = propensity,
      outcome_mean = if (is.null(outcome_mean)) NULL else .cc_as_numeric_vector(outcome_mean, "outcome_mean")
    )
    if (is.null(outcome_mean)) {
      stop(
        "Overlap-targeted diagnostics require `outcome_mean` or enough nuisance information to infer it.",
        call. = FALSE
      )
    }
    r_target <- .cc_r_pseudo_outcome(treatment, outcome, outcome_mean, propensity)
    effective_weight <- sample_weight * r_target$weights
    .cc_validate_nonnegative_weights(effective_weight, "overlap_effective_weight")
    overlap_result <- .cc_build_target_result(
      predictions = predictions,
      pseudo_outcome = r_target$target,
      sample_weight = effective_weight,
      method = curve_method,
      method_options = method_options,
      fold_ids = fold_ids,
      confidence_level = confidence_level,
      target_population = "overlap",
      comparison_predictions = comparison_predictions
    )
  }

  primary <- if (target_population == "overlap") overlap_result else dr_result
  if (is.null(primary)) {
    stop("Failed to construct the requested diagnostics target.", call. = FALSE)
  }
  structure(
    list(
      estimate = primary$estimate,
      plugin_estimate = primary$plugin_estimate,
      standard_error = primary$standard_error,
      confidence_interval = primary$confidence_interval,
      curve_predictions = primary$curve_predictions,
      curve_estimates = primary$curve_estimates,
      curve_method = primary$curve_method,
      jackknife_folds = primary$jackknife_folds,
      target_population = target_population,
      fold_estimates = primary$fold_estimates,
      comparison_estimate = primary$comparison_estimate,
      comparison_standard_error = primary$comparison_standard_error,
      overlap_diagnostics = overlap,
      dr_result = dr_result,
      overlap_result = overlap_result
    ),
    class = "causal_calibration_diagnostics"
  )
}

#' @export
summary.causal_calibration_target_result <- function(object, ...) {
  summary <- list(
    target_population = object$target_population,
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
summary.causal_calibration_diagnostics <- function(object, ...) {
  summary <- list(
    target_population = object$target_population,
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
  if (!is.null(object$overlap_diagnostics)) {
    summary$overlap <- summary(object$overlap_diagnostics)
  }
  if (object$target_population == "both") {
    summary$dr_result <- if (is.null(object$dr_result)) NULL else summary(object$dr_result)
    summary$overlap_result <- if (is.null(object$overlap_result)) NULL else summary(object$overlap_result)
  }
  summary
}

#' @export
print.causal_calibration_diagnostics <- function(x, ...) {
  cat("<causal_calibration_diagnostics>\n")
  cat(sprintf("  target_population: %s\n", x$target_population))
  cat(sprintf("  estimate: %.6f\n", x$estimate))
  cat(sprintf("  standard_error: %.6f\n", x$standard_error))
  invisible(x)
}

#' @export
plot.causal_calibration_diagnostics <- function(x, ...) {
  graphics::plot(
    x$curve_predictions,
    x$curve_estimates,
    type = "l",
    xlab = "Prediction",
    ylab = "Estimated calibration curve",
    ...
  )
  if (x$target_population == "both") {
    if (!is.null(x$dr_result)) {
      graphics::lines(x$dr_result$curve_predictions, x$dr_result$curve_estimates, col = "steelblue")
    }
    if (!is.null(x$overlap_result)) {
      graphics::lines(x$overlap_result$curve_predictions, x$overlap_result$curve_estimates, col = "firebrick")
    }
    graphics::legend(
      "topleft",
      legend = c("primary", "dr", "overlap"),
      col = c("black", "steelblue", "firebrick"),
      lty = 1,
      bty = "n"
    )
  }
}
