\name{fit_calibrator}
\alias{fit_calibrator}
\title{Fit a causal calibration map}
\usage{
fit_calibrator(
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
  clip = 1e-06,
  method_options = list()
)
}
\arguments{
\item{predictions}{Numeric vector of treatment-effect predictions to calibrate.}
\item{treatment}{Numeric binary vector of treatment assignments.}
\item{outcome}{Numeric vector of observed outcomes.}
\item{loss}{Calibration loss, either \code{"dr"} or \code{"r"}.}
\item{method}{Calibration backend: \code{"isotonic"}, \code{"smooth_isotonic"}, \code{"linear"}, or \code{"histogram"}.}
\item{mu0}{Numeric vector of estimated control outcome regressions.}
\item{mu1}{Numeric vector of estimated treated outcome regressions.}
\item{outcome_mean}{Numeric vector of estimated marginal outcome means, used for \code{loss = "r"}.}
\item{propensity}{Numeric vector of estimated propensity scores.}
\item{sample_weight}{Optional numeric vector of observation weights.}
\item{clip}{Propensity clipping threshold.}
\item{method_options}{Optional named list of backend-specific settings.}
}
\value{
A fitted calibrator object of class \code{"causal_calibrator"}.
}
\description{
Fits a calibration model for heterogeneous treatment effect predictions using
either doubly robust or residualized R-loss targets.
}
\details{
Use \code{loss = "dr"} when you want calibration targeting the original study
population and can tolerate inverse-propensity weighting. Use \code{loss = "r"}
when you want a calibration target that emphasizes the overlap region via the
R-loss weighting and may be more stable under weak overlap, with the caveat
that the calibrated target becomes overlap-weighted rather than the original
study population.
}
