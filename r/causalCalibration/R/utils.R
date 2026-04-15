.cc_as_numeric_vector <- function(x, name) {
  if (length(x) == 0) {
    stop(sprintf("`%s` must contain at least one value.", name), call. = FALSE)
  }
  values <- as.numeric(x)
  if (any(!is.finite(values))) {
    stop(sprintf("`%s` must contain only finite numeric values.", name), call. = FALSE)
  }
  values
}

.cc_validate_same_length <- function(length_expected, ...) {
  inputs <- list(...)
  for (name in names(inputs)) {
    if (length(inputs[[name]]) != length_expected) {
      stop(
        sprintf("`%s` must have length %s, received %s.", name, length_expected, length(inputs[[name]])),
        call. = FALSE
      )
    }
  }
}

.cc_validate_binary <- function(x, name) {
  if (any(!(x %in% c(0, 1)))) {
    stop(sprintf("`%s` must be binary with values in {0, 1}.", name), call. = FALSE)
  }
}

.cc_validate_method <- function(method) {
  valid <- c("histogram", "isotonic", "linear", "monotone_spline")
  if (!(method %in% valid)) {
    stop(
      sprintf("`method` must be one of %s.", paste(sprintf("'%s'", valid), collapse = ", ")),
      call. = FALSE
    )
  }
  method
}

.cc_validate_min_unique_scores <- function(x, method) {
  n_unique <- length(unique(round(x, 12)))
  if (n_unique < 2L) {
    stop("Calibration requires at least two distinct prediction values.", call. = FALSE)
  }
  if (method == "monotone_spline" && n_unique < 4L) {
    stop("`monotone_spline` requires at least four distinct prediction values.", call. = FALSE)
  }
}

.cc_clip_propensity <- function(x, clip) {
  if (clip < 0 || clip >= 0.5) {
    stop("`clip` must be in [0, 0.5).", call. = FALSE)
  }
  clipped <- pmin(pmax(x, clip), 1 - clip)
  if (any(clipped <= 0 | clipped >= 1)) {
    stop("Propensity scores must lie strictly between 0 and 1 after clipping.", call. = FALSE)
  }
  clipped
}

.cc_weighted_mean <- function(x, w) {
  total_weight <- sum(w)
  if (total_weight <= 0) {
    stop("Weights must sum to a positive value.", call. = FALSE)
  }
  sum(x * w) / total_weight
}

.cc_validate_nonnegative_weights <- function(w, name = "sample_weight") {
  if (any(w < 0)) {
    stop(sprintf("`%s` must contain only nonnegative values.", name), call. = FALSE)
  }
  if (sum(w) <= 0) {
    stop(sprintf("`%s` must sum to a positive value.", name), call. = FALSE)
  }
}

.cc_weighted_effective_sample_size <- function(w) {
  total <- sum(w)
  squared <- sum(w^2)
  if (total <= 0 || squared <= 0) {
    return(0)
  }
  total^2 / squared
}

.cc_order_statistic_median <- function(x) {
  x <- sort(x)
  x[(length(x) + 1) %/% 2]
}

.cc_balanced_folds <- function(n, k) {
  if (k < 2) {
    stop("`jackknife_folds` must be at least 2.", call. = FALSE)
  }
  ((seq_len(n) - 1L) %% k) + 1L
}

.cc_validate_fold_ids <- function(fold_ids, n) {
  folds <- as.integer(fold_ids)
  .cc_validate_same_length(n, fold_ids = folds)
  if (any(folds < 1L)) {
    stop("`fold_ids` must contain positive integers.", call. = FALSE)
  }
  observed <- sort(unique(folds))
  expected <- seq_len(max(observed))
  if (!identical(observed, expected)) {
    stop("`fold_ids` must cover folds consecutively starting at 1 with no gaps.", call. = FALSE)
  }
  folds
}

.cc_mapping_grid <- function(x, n_points = 200L) {
  minimum <- min(x)
  maximum <- max(x)
  if (minimum == maximum) {
    return(minimum)
  }
  seq(minimum, maximum, length.out = n_points)
}

.cc_normal_quantile <- function(level) {
  if (level <= 0 || level >= 1) {
    stop("`confidence_level` must lie in (0, 1).", call. = FALSE)
  }
  stats::qnorm(0.5 + level / 2)
}

.cc_as_matrix <- function(x, name) {
  if (is.matrix(x)) {
    mat <- x * 1
  } else if (is.data.frame(x)) {
    mat <- as.matrix(x)
  } else if (is.atomic(x)) {
    mat <- matrix(as.numeric(x), ncol = 1L)
  } else {
    stop(sprintf("`%s` must be a vector, matrix, or data frame.", name), call. = FALSE)
  }
  if (nrow(mat) == 0L || ncol(mat) == 0L) {
    stop(sprintf("`%s` must have at least one row and one column.", name), call. = FALSE)
  }
  if (any(!is.finite(mat))) {
    stop(sprintf("`%s` must contain only finite numeric values.", name), call. = FALSE)
  }
  mat
}

.cc_validate_oof_alignment <- function(predictions, fold_predictions, fold_ids, tolerance = 1e-8) {
  max_abs_error <- 0
  for (index in seq_along(predictions)) {
    oof_value <- fold_predictions[index, fold_ids[index]]
    error <- abs(predictions[index] - oof_value)
    max_abs_error <- max(max_abs_error, error)
    if (error > tolerance) {
      stop(
        "Pooled out-of-fold `predictions` do not match `fold_predictions` at the designated `fold_ids`.",
        call. = FALSE
      )
    }
  }
  list(max_abs_error = max_abs_error)
}

.cc_infer_outcome_mean <- function(mu0 = NULL, mu1 = NULL, propensity = NULL, outcome_mean = NULL) {
  if (!is.null(outcome_mean)) {
    return(outcome_mean)
  }
  if (is.null(mu0) || is.null(mu1) || is.null(propensity)) {
    return(NULL)
  }
  (1 - propensity) * mu0 + propensity * mu1
}
