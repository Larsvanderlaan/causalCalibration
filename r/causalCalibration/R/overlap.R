#' Assess treatment overlap from propensity scores
#'
#' Applies the package's default overlap screen and recommends a calibration loss.
#'
#' @param treatment Numeric binary vector of treatment assignments.
#' @param propensity Numeric vector of propensity scores.
#' @param sample_weight Optional numeric vector of observation weights.
#' @param clip Propensity clipping threshold.
#'
#' @return An object of class `"causal_overlap_diagnostics"`.
#' @export
assess_overlap <- function(
  treatment,
  propensity,
  sample_weight = NULL,
  clip = 1e-6
) {
  treatment <- .cc_as_numeric_vector(treatment, "treatment")
  propensity_raw <- .cc_as_numeric_vector(propensity, "propensity")
  .cc_validate_same_length(length(treatment), propensity = propensity_raw)
  .cc_validate_binary(treatment, "treatment")
  sample_weight <- if (is.null(sample_weight)) rep(1, length(treatment)) else .cc_as_numeric_vector(sample_weight, "sample_weight")
  .cc_validate_same_length(length(treatment), sample_weight = sample_weight)
  .cc_validate_nonnegative_weights(sample_weight)
  propensity_clipped <- .cc_clip_propensity(propensity_raw, clip)

  ipw_weights <- sample_weight * ((treatment / propensity_clipped) + ((1 - treatment) / (1 - propensity_clipped)))
  overlap_weights <- sample_weight * propensity_clipped * (1 - propensity_clipped)
  clipped_fraction <- mean(abs(propensity_raw - propensity_clipped) > 1e-12)
  min_propensity <- min(propensity_raw)
  max_propensity <- max(propensity_raw)
  fraction_below_005 <- mean(propensity_raw < 0.05)
  fraction_above_095 <- mean(propensity_raw > 0.95)
  fraction_below_010 <- mean(propensity_raw < 0.10)
  fraction_above_090 <- mean(propensity_raw > 0.90)
  ipw_ess <- .cc_weighted_effective_sample_size(ipw_weights)
  overlap_ess <- .cc_weighted_effective_sample_size(overlap_weights)

  severe <- clipped_fraction > 0.02 || (ipw_ess / length(propensity_raw)) < 0.25
  weak <- min_propensity < 0.05 || max_propensity > 0.95 || severe
  severity <- if (severe) "severe" else if (weak) "weak" else "adequate"
  recommended_loss <- if (severity == "adequate") "dr" else "r"

  structure(
    list(
      min_propensity = min_propensity,
      max_propensity = max_propensity,
      fraction_below_005 = fraction_below_005,
      fraction_above_095 = fraction_above_095,
      fraction_below_010 = fraction_below_010,
      fraction_above_090 = fraction_above_090,
      clipped_fraction = clipped_fraction,
      ipw_effective_sample_size = ipw_ess,
      overlap_effective_sample_size = overlap_ess,
      severity = severity,
      recommended_loss = recommended_loss,
      clip = clip,
      n_obs = length(propensity_raw)
    ),
    class = "causal_overlap_diagnostics"
  )
}

.cc_overlap_messages <- function(overlap) {
  if (is.null(overlap) || overlap$severity == "adequate") {
    return(character())
  }
  messages <- "The package's default overlap screen flagged weak overlap; consider `loss = \"r\"` for overlap-weighted calibration."
  if (overlap$clipped_fraction > 0.02) {
    messages <- c(
      messages,
      "A nontrivial fraction of propensities were clipped; DR-targeted calibration may be unstable."
    )
  }
  if ((overlap$ipw_effective_sample_size / max(overlap$n_obs, 1)) < 0.25) {
    messages <- c(
      messages,
      "The IPW effective sample size is small relative to n, which suggests unstable original-population weighting."
    )
  }
  messages
}

.cc_warn_overlap <- function(overlap, loss) {
  messages <- .cc_overlap_messages(overlap)
  if (length(messages) == 0L) {
    return(invisible(NULL))
  }
  for (message in messages) {
    if (overlap$recommended_loss != loss || overlap$severity == "severe") {
      warning(message, call. = FALSE, immediate. = TRUE)
    }
  }
  invisible(NULL)
}

#' @export
summary.causal_overlap_diagnostics <- function(object, ...) {
  unclass(object)
}

#' @export
print.causal_overlap_diagnostics <- function(x, ...) {
  cat("<causal_overlap_diagnostics>\n")
  cat(sprintf("  severity: %s\n", x$severity))
  cat(sprintf("  recommended_loss: %s\n", x$recommended_loss))
  cat(sprintf("  min_propensity: %.4f\n", x$min_propensity))
  cat(sprintf("  max_propensity: %.4f\n", x$max_propensity))
  invisible(x)
}

#' @export
plot.causal_overlap_diagnostics <- function(x, ...) {
  values <- c(
    x$fraction_below_005,
    x$fraction_below_010,
    x$fraction_above_090,
    x$fraction_above_095,
    x$clipped_fraction
  )
  graphics::barplot(
    values,
    names.arg = c("<0.05", "<0.10", ">0.90", ">0.95", "clipped"),
    ylab = "Fraction",
    main = sprintf("Overlap diagnostics (%s)", x$severity),
    ...
  )
}
