.cc_dr_pseudo_outcome <- function(treatment, outcome, mu0, mu1, propensity) {
  mu_observed <- ifelse(treatment == 1, mu1, mu0)
  propensity_observed <- ifelse(treatment == 1, propensity, 1 - propensity)
  mu1 - mu0 + ((2 * treatment - 1) / propensity_observed) * (outcome - mu_observed)
}

.cc_r_pseudo_outcome <- function(treatment, outcome, outcome_mean, propensity) {
  residual <- treatment - propensity
  if (any(residual == 0)) {
    stop("R-loss residualized treatment is zero; adjust propensity clipping.", call. = FALSE)
  }
  list(target = (outcome - outcome_mean) / residual, weights = residual^2)
}

#' Fit a causal calibration map
#'
#' Fits a calibration model for heterogeneous treatment effect predictions using
#' either doubly robust (`loss = "dr"`) or residualized R-loss (`loss = "r"`)
#' targets.
#'
#' @param predictions Numeric vector of treatment-effect predictions to calibrate.
#' @param treatment Numeric binary vector of treatment assignments.
#' @param outcome Numeric vector of observed outcomes.
#' @param loss Calibration loss, either `"dr"` or `"r"`.
#' @param method Calibration backend: `"isotonic"`, `"smooth_isotonic"`,
#'   `"linear"`, or `"histogram"`.
#' @param mu0 Numeric vector of estimated control outcome regressions.
#' @param mu1 Numeric vector of estimated treated outcome regressions.
#' @param outcome_mean Numeric vector of estimated marginal outcome means, used
#'   for `loss = "r"`.
#' @param propensity Numeric vector of estimated propensity scores.
#' @param sample_weight Optional numeric vector of observation weights.
#' @param clip Propensity clipping threshold.
#' @param method_options Optional named list of backend-specific settings.
#'
#' @return A fitted calibrator object of class `"causal_calibrator"`.
#' @export
fit_calibrator <- function(
  predictions,
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
  predictions <- .cc_as_numeric_vector(predictions, "predictions")
  treatment <- .cc_as_numeric_vector(treatment, "treatment")
  outcome <- .cc_as_numeric_vector(outcome, "outcome")
  .cc_validate_same_length(length(predictions), treatment = treatment, outcome = outcome)
  .cc_validate_binary(treatment, "treatment")
  sample_weight <- if (is.null(sample_weight)) rep(1, length(predictions)) else .cc_as_numeric_vector(sample_weight, "sample_weight")
  .cc_validate_same_length(length(predictions), sample_weight = sample_weight)
  propensity <- if (is.null(propensity)) NULL else .cc_clip_propensity(.cc_as_numeric_vector(propensity, "propensity"), clip)

  if (loss == "dr") {
    if (is.null(mu0) || is.null(mu1) || is.null(propensity)) {
      stop("DR loss requires `mu0`, `mu1`, and `propensity`.", call. = FALSE)
    }
    mu0 <- .cc_as_numeric_vector(mu0, "mu0")
    mu1 <- .cc_as_numeric_vector(mu1, "mu1")
    .cc_validate_same_length(length(predictions), mu0 = mu0, mu1 = mu1, propensity = propensity)
    target <- .cc_dr_pseudo_outcome(treatment, outcome, mu0, mu1, propensity)
    effective_weight <- sample_weight
  } else if (loss == "r") {
    if (is.null(outcome_mean) || is.null(propensity)) {
      stop("R loss requires `outcome_mean` and `propensity`.", call. = FALSE)
    }
    outcome_mean <- .cc_as_numeric_vector(outcome_mean, "outcome_mean")
    .cc_validate_same_length(length(predictions), outcome_mean = outcome_mean, propensity = propensity)
    r_fit <- .cc_r_pseudo_outcome(treatment, outcome, outcome_mean, propensity)
    target <- r_fit$target
    effective_weight <- sample_weight * r_fit$weights
  } else {
    stop("`loss` must be either 'dr' or 'r'.", call. = FALSE)
  }

  model <- .cc_fit_backend(method, predictions, target, effective_weight, method_options = method_options)
  fitted_values <- .cc_predict_backend(model, predictions)
  structure(
    list(
      loss = loss,
      method = method,
      model = model,
      clip = clip,
      n_obs = length(predictions),
      method_options = method_options,
      training_predictions = predictions,
      fitted_values = fitted_values,
      effective_weights = effective_weight
    ),
    class = "causal_calibrator"
  )
}

#' @export
predict.causal_calibrator <- function(object, newdata, ...) {
  values <- .cc_as_numeric_vector(newdata, "newdata")
  output <- .cc_predict_backend(object$model, values)
  if (length(output) == 1 && length(newdata) == 1) {
    return(output[[1]])
  }
  output
}

#' @export
summary.causal_calibrator <- function(object, ...) {
  list(
    loss = object$loss,
    method = object$method,
    n_obs = object$n_obs,
    clip = object$clip,
    method_options = object$method_options
  )
}

#' @export
print.causal_calibrator <- function(x, ...) {
  cat("<causal_calibrator>\n")
  cat(sprintf("  loss: %s\n", x$loss))
  cat(sprintf("  method: %s\n", x$method))
  cat(sprintf("  n_obs: %s\n", x$n_obs))
  invisible(x)
}

#' @export
plot.causal_calibrator <- function(x, ...) {
  grid <- .cc_mapping_grid(x$training_predictions)
  calibrated <- .cc_predict_backend(x$model, grid)
  plot(grid, calibrated, type = "l", xlab = "Prediction", ylab = "Calibrated prediction", ...)
}
