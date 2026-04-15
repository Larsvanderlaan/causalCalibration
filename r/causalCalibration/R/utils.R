.cc_as_numeric_vector <- function(x, name) {
  if (length(x) == 0) {
    stop(sprintf("`%s` must contain at least one value.", name), call. = FALSE)
  }
  as.numeric(x)
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
    return(x * 1)
  }
  if (is.data.frame(x)) {
    return(as.matrix(x))
  }
  if (is.atomic(x)) {
    return(matrix(as.numeric(x), ncol = 1))
  }
  stop(sprintf("`%s` must be a vector, matrix, or data frame.", name), call. = FALSE)
}
