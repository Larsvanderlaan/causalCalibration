\name{diagnose_calibration}
\alias{diagnose_calibration}
\title{Diagnose calibration error with a doubly robust estimator}
\usage{
diagnose_calibration(
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
  clip = 1e-06,
  confidence_level = 0.95
)
}
\arguments{
\item{predictions}{Numeric vector of treatment-effect predictions to evaluate.}
\item{treatment}{Numeric binary vector of treatment assignments.}
\item{outcome}{Numeric vector of observed outcomes.}
\item{mu0}{Numeric vector of estimated control outcome regressions.}
\item{mu1}{Numeric vector of estimated treated outcome regressions.}
\item{propensity}{Numeric vector of estimated propensity scores.}
\item{sample_weight}{Optional numeric vector of observation weights.}
\item{comparison_predictions}{Optional second prediction vector used for before/after comparisons.}
\item{curve_method}{Calibration curve backend used for diagnostics.}
\item{method_options}{Optional named list of backend-specific settings.}
\item{fold_ids}{Optional integer vector of jackknife fold assignments.}
\item{jackknife_folds}{Number of deterministic balanced folds when \code{fold_ids} is not supplied.}
\item{clip}{Propensity clipping threshold.}
\item{confidence_level}{Confidence level for the normal-approximation interval.}
}
\value{
A diagnostics object of class \code{"causal_calibration_diagnostics"}.
}
\description{
Estimates a calibration curve and L2 calibration error using the doubly robust
calibration-error estimator from Xu and Yadlowsky (2022), together with
deterministic K-fold jackknife standard errors.
}
\details{
Diagnostics operate on prediction vectors and nuisance estimates rather than
learner objects. Jackknife standard errors use deterministic folds, with
\code{jackknife_folds = 100L} by default when fold IDs are not supplied.
Debiasing uses exact leave-one-out updates for \code{linear}, cheap within-bin
leave-one-out corrections for \code{histogram}, and K-fold out-of-fold refits
for \code{isotonic} and \code{smooth_isotonic}.
}
