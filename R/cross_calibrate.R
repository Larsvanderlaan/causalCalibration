

#' Second-stage function that computes cross-calibrated predictions. Requires first applying causalCalibration with pooled out-of-fold cross-fitted predictions.
#' @param output An output list from the causalCalibrate function obtained from passing in the pooled out-of-fold predictions of the cross-fitted uncalibrated predictors.
#' @param new_tau_mat An n by k matrix of n row-wise stacked predictions from k cross-fitted uncalibrated predictors.
#' These predictions can be for observations at which we wish to obtain calibrated predictions (e.g., out-of-sample).
#' Each column should correspond with the treatment effect predictions from one of the k fold-specific uncalibrated predictors.
#' @returns A vector of n calibrated predictions obtained by taking pointwise medians of each of the fold-specific calibrated predictors.
#' @export
cross_calibrate <- function(output, new_tau_mat) {
  calibration_function <- output$calibration_function
  tau_mat_cal <- apply(new_tau_mat, 2, calibration_function)
  # definition of median is important for theory. It must be an order statistic, so no averaging if ties.
  tau_cal <- as.vector(apply(tau_mat_cal, 1, quantile, type = 1, probs = 0.5))
  return(tau_cal)
}

