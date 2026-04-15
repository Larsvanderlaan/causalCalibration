\name{fit_cross_calibrator}
\alias{fit_cross_calibrator}
\title{Fit a cross-calibrator}
\usage{
fit_cross_calibrator(
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
  clip = 1e-06,
  method_options = list()
)
}
\arguments{
\item{predictions}{Numeric vector of pooled out-of-fold predictions.}
\item{fold_predictions}{Numeric matrix with one column per fold-specific predictor.}
\item{treatment}{Numeric binary vector of treatment assignments.}
\item{outcome}{Numeric vector of observed outcomes.}
\item{loss}{Calibration loss, either \code{"dr"} or \code{"r"}.}
\item{method}{Calibration backend.}
\item{mu0}{Numeric vector of estimated control outcome regressions.}
\item{mu1}{Numeric vector of estimated treated outcome regressions.}
\item{outcome_mean}{Numeric vector of estimated marginal outcome means, used for \code{loss = "r"}.}
\item{propensity}{Numeric vector of estimated propensity scores.}
\item{sample_weight}{Optional numeric vector of observation weights.}
\item{clip}{Propensity clipping threshold.}
\item{method_options}{Optional named list of backend-specific settings.}
}
\value{
A fitted object of class \code{"causal_cross_calibrator"}.
}
\description{
Fits a calibrator on pooled out-of-fold predictions and stores the expected
number of fold-specific prediction columns for prediction-time cross-calibration.
}
\details{
The argument \code{predictions} should contain pooled out-of-fold predictions
used to fit the calibration map. The argument \code{fold_predictions} should be
an \eqn{n \times K} matrix whose columns correspond to fold-specific predictors.
Prediction from the fitted object applies the calibration map column-wise and
aggregates with the order-statistic median.
}
