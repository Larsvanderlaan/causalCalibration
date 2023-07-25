
#' Causal calibration of conditional average treatment effect (CATE) predictors
#'
#' We recommend passing in pooled out-of-fold estimates obtained from cross-fitted nuisance function estimators.
#' To implement cross-calibration, the argument `tau` should be the pooled out-of-fold treatment effect predictions from the cross-fitted predictors.
#' The output can then be passed to the function `cross_calibrate` to obtain the final cross-calibrated predictions.
#' Suppose we observe `n` realizations of the data-structure `(W, A, Y)`
#' where:
#' - `W` is a possibly high dimensional vector of baseline variables (i.e. the context)
#' - `A` is a binary treatment taking values in {0,1} (i.e. the action)
#' - `Y` is an arbitrary outcome that captures the effectiveness of the treatment `A` (i.e. the reward)
#'
#' Let `w -> tau(w)` be a treatment effect predictor that maps a given context `w` to a treatment effect score `tau(w)`.
#'
#' The methods in this package calibrate `tau` so that the resulting calibrated predictor `tau*`
#'satisfies `tau*(w) = E[Y(1) - Y(0)|tau*(W) = tau*(w)]` in an L^2(P_W) sense up to an error of `n^(-1/3)`.
#'
#' In other words, the conditional average treatment effect of individuals with the same predicted calibrated treatment effectiveness score `tau*(w)`
#' is the treatment effectiveness score itself, `tau*(w)`.
#'

#' This method requires estimates of the propensity score and outcome regression evaluated at each of the observations.
#' @param tau A vector of initial treatment effect predictions for observations in the calibration set.
#' @param A A binary vector of treatment assignments for observations in the calibration set.
#' @param Y A vector of outcomes for observations in the calibration set.
#' @param pA1 (Optional) A vector of estimates of either the propensity score `P(A=1|W)` for the calibration set.
#' @param EY1 (Optional) A vector of estimates of either the treatment-specific outcome regression `E[Y|A=1, W]` for the calibration set.
#' @param EY0 (Optional) A vector of estimates of either the treatment-specific outcome regression `E[Y|A=0, W]` for the calibration set.
#' @param weights An optional vector of observations weights to incorporate in the calibration procedure.
#' @param tau_pred (optional) A vector of initial treatment effect predictions to calibrate.
#' @returns tau_calibrated: The calibrated treatment effect predictions obtained by applying the calibration mapping to `tau_pred`.
#' @returns calibration_function: A calibration function that can be used to map the initial treatment effect predictions (`tau`) to calibrated predictions.
#' @returns iso_reg_fit: The internal isotonic calibration fit that the calibration_function is derived from.
#' @import mgcv
#' @export
causalCalibrate <- function(tau, A, Y, EY1, EY0, pA1, weights = rep(1, length(tau)), tau_pred = tau) {


  EY <- ifelse(A==1, EY1, EY0)
  pA <- ifelse(A==1, pA1, 1-pA1)

  # pseudo_outcome used for calibration
  pseudo_outcome <- EY1 - EY0 + (2*A-1)/pA * (Y - EY)

  # causal isotonic calibration
  fit_iso <- isoreg(tau, pseudo_outcome)

  # correct poor boundary behavior, an artifact of isotonic regression.
  # This correction is adhoc and there are probably more principled ways to do this.

  ymin <- sort(fit_iso$yf)[1]
  ymin2 <- sort(fit_iso$yf)[2]
  fit_iso$yf[fit_iso$yf==ymin] <- ymin2
  ymax <- rev(sort(fit_iso$yf))[1]
  ymax2 <- rev(sort(fit_iso$yf))[2]
  fit_iso$yf[fit_iso$yf==ymax] <- ymax2

  # Step function from iso fit
  calibration_function <- as.stepfun(fit_iso)

  return(list( tau_calibrated = calibration_function(tau_pred), calibration_function = calibration_function, iso_reg_fit = fit_iso  ))
}




