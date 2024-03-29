% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cross_calibrate.R
\name{cross_calibrate}
\alias{cross_calibrate}
\title{Second-stage function that computes cross-calibrated predictions. Requires first applying causalCalibration with pooled out-of-fold cross-fitted predictions.}
\usage{
cross_calibrate(output, new_tau_mat)
}
\arguments{
\item{output}{An output list from the causalCalibrate function obtained from passing in the pooled out-of-fold predictions of the cross-fitted uncalibrated predictors.}

\item{new_tau_mat}{An n by k matrix of n row-wise stacked predictions from k cross-fitted uncalibrated predictors.
These predictions can be for observations at which we wish to obtain calibrated predictions (e.g., out-of-sample).
Each column should correspond with the treatment effect predictions from one of the k fold-specific uncalibrated predictors.}
}
\value{
A vector of n calibrated predictions obtained by taking pointwise medians of each of the fold-specific calibrated predictors.
}
\description{
Second-stage function that computes cross-calibrated predictions. Requires first applying causalCalibration with pooled out-of-fold cross-fitted predictions.
}
