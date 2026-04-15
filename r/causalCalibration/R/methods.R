.cc_py_modules <- new.env(parent = emptyenv())

.cc_import_lightgbm <- function() {
  module <- .cc_py_modules$lightgbm
  if (!is.null(module)) {
    return(module)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package `reticulate` is required for `method = \"isotonic\"`.", call. = FALSE)
  }
  .cc_py_modules$lightgbm <- tryCatch(
    reticulate::import("lightgbm", delay_load = FALSE),
    error = function(error) {
      stop(
        paste(
          "Python package `lightgbm` is required for `method = \"isotonic\"`.",
          "Install it in the active reticulate environment."
        ),
        call. = FALSE
      )
    }
  )
  .cc_py_modules$lightgbm
}

.cc_predict_lightgbm_isotonic <- function(model, x) {
  clipped <- pmin(pmax(as.numeric(x), model$score_min), model$score_max)
  as.numeric(model$estimator$predict(matrix(clipped, ncol = 1L)))
}

.cc_predict_linear <- function(model, x) {
  model$intercept + model$slope * as.numeric(x)
}

.cc_predict_histogram <- function(model, x) {
  values <- as.numeric(x)
  vapply(
    values,
    function(value) {
      index <- findInterval(value, model$lower_bounds)
      index <- max(1L, min(index, length(model$values)))
      model$values[[index]]
    },
    numeric(1)
  )
}

.cc_weighted_quantile_breaks <- function(x, weights, n_bins) {
  ord <- order(x)
  x <- x[ord]
  weights <- weights[ord]
  total_weight <- sum(weights)
  targets <- seq_len(n_bins - 1L) / n_bins * total_weight
  breaks <- numeric()
  cumulative <- 0
  target_index <- 1L
  for (index in seq_along(x)) {
    cumulative <- cumulative + weights[[index]]
    while (target_index <= length(targets) && cumulative >= targets[[target_index]]) {
      if (length(breaks) == 0L || x[[index]] > breaks[[length(breaks)]]) {
        breaks <- c(breaks, x[[index]])
      }
      target_index <- target_index + 1L
    }
  }
  breaks
}

.cc_fit_histogram <- function(x, y, weights, n_bins = 10L) {
  if (n_bins < 1L) {
    stop("`n_bins` must be at least 1.", call. = FALSE)
  }
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]
  weights <- weights[ord]
  breaks <- .cc_weighted_quantile_breaks(x, weights, n_bins)
  lower_bounds <- c(x[[1]], breaks[breaks > x[[1]]])
  bin_ids <- findInterval(x, lower_bounds)
  bin_ids[bin_ids < 1L] <- 1L
  values <- numeric(length(lower_bounds))
  upper_bounds <- numeric(length(lower_bounds))
  overall_mean <- .cc_weighted_mean(y, weights)
  for (index in seq_along(lower_bounds)) {
    in_bin <- which(bin_ids == index)
    if (length(in_bin) == 0L) {
      values[[index]] <- if (index == 1L) overall_mean else values[[index - 1L]]
      upper_bounds[[index]] <- lower_bounds[[index]]
    } else {
      values[[index]] <- .cc_weighted_mean(y[in_bin], weights[in_bin])
      upper_bounds[[index]] <- max(x[in_bin])
    }
  }
  structure(
    list(
      kind = "histogram",
      lower_bounds = lower_bounds,
      upper_bounds = upper_bounds,
      values = values
    ),
    class = "cc_histogram_model"
  )
}

.cc_fit_linear <- function(x, y, weights) {
  x_mean <- .cc_weighted_mean(x, weights)
  y_mean <- .cc_weighted_mean(y, weights)
  denominator <- sum(weights * (x - x_mean)^2)
  if (denominator == 0) {
    return(
      structure(
        list(kind = "linear", intercept = y_mean, slope = 0),
        class = "cc_linear_model"
      )
    )
  }
  numerator <- sum(weights * (x - x_mean) * (y - y_mean))
  slope <- max(0, numerator / denominator)
  structure(
    list(
      kind = "linear",
      intercept = y_mean - slope * x_mean,
      slope = slope
    ),
    class = "cc_linear_model"
  )
}

.cc_fit_isotonic <- function(
  x,
  y,
  weights,
  max_depth = 20L,
  min_child_samples = 10L,
  learning_rate = 1,
  n_estimators = 1L
) {
  lightgbm <- .cc_import_lightgbm()
  estimator <- lightgbm$LGBMRegressor(
    objective = "regression",
    n_estimators = as.integer(n_estimators),
    learning_rate = as.numeric(learning_rate),
    max_depth = as.integer(max_depth),
    min_child_samples = as.integer(min_child_samples),
    monotone_constraints = list(1L),
    num_leaves = 31L,
    subsample = 1,
    colsample_bytree = 1,
    reg_lambda = 0,
    verbosity = -1L,
    random_state = 0L
  )
  estimator$fit(matrix(as.numeric(x), ncol = 1L), as.numeric(y), sample_weight = as.numeric(weights))
  structure(
    list(
      kind = "isotonic",
      estimator = estimator,
      score_min = min(x),
      score_max = max(x),
      metadata = list(
        max_depth = as.integer(max_depth),
        min_child_samples = as.integer(min_child_samples),
        learning_rate = as.numeric(learning_rate),
        n_estimators = as.integer(n_estimators)
      )
    ),
    class = "cc_lightgbm_isotonic_model"
  )
}

.cc_monotone_spline_max_internal_knots <- 6L
.cc_monotone_spline_derivative_degree <- 2L
.cc_monotone_spline_penalty <- 1e-3

.cc_choose_monotone_spline_knots <- function(scores_scaled, max_internal_knots, degree) {
  unique_scores <- sort(unique(scores_scaled))
  max_allowed <- max(0L, length(unique_scores) - degree - 1L)
  n_internal <- min(max_internal_knots, max_allowed)
  if (n_internal <= 0L) {
    internal <- numeric()
  } else {
    probs <- seq(0, 1, length.out = n_internal + 2L)[-c(1L, n_internal + 2L)]
    internal <- unique(as.numeric(stats::quantile(scores_scaled, probs = probs, type = 7)))
    internal <- internal[internal > 1e-8 & internal < 1 - 1e-8]
  }
  c(rep(0, degree + 1L), internal, rep(1, degree + 1L))
}

.cc_integrated_bspline_design <- function(scores_scaled, knots, degree) {
  x <- pmax(pmin(as.numeric(scores_scaled), 1), 0)
  grid <- sort(unique(c(0, x)))
  basis <- splines::splineDesign(knots = knots, x = grid, ord = degree + 1L, outer.ok = TRUE)
  n_basis <- ncol(basis)
  if (n_basis <= 0L) {
    stop("Invalid spline specification: expected at least one basis function.", call. = FALSE)
  }
  integ <- matrix(0, nrow = length(grid), ncol = n_basis)
  if (length(grid) > 1L) {
    dx <- diff(grid)
    for (j in seq_len(n_basis)) {
      for (i in 2:length(grid)) {
        integ[i, j] <- integ[i - 1L, j] + 0.5 * dx[i - 1L] * (basis[i - 1L, j] + basis[i, j])
      }
    }
  }
  integ[match(x, grid), , drop = FALSE]
}

.cc_evaluate_monotone_spline <- function(scores_scaled, knots, coef, degree, intercept) {
  basis <- .cc_integrated_bspline_design(scores_scaled, knots = knots, degree = degree)
  as.numeric(intercept + basis %*% coef)
}

.cc_fit_monotone_spline <- function(
  x,
  y,
  weights,
  max_internal_knots = .cc_monotone_spline_max_internal_knots,
  basis_degree = .cc_monotone_spline_derivative_degree + 1L,
  penalty = .cc_monotone_spline_penalty
) {
  .cc_validate_min_unique_scores(x, "monotone_spline")
  y_min <- min(y)
  y_max <- max(y)
  if (length(unique(round(x, 12))) <= 1L || isTRUE(all.equal(y_min, y_max))) {
    return(.cc_fit_linear(x, y, weights))
  }
  score_min <- min(x)
  score_max <- max(x)
  score_scale <- score_max - score_min
  if (isTRUE(all.equal(score_scale, 0)) || length(unique(round(x, 12))) < 4L) {
    return(.cc_fit_linear(x, y, weights))
  }
  derivative_degree <- max(0L, as.integer(basis_degree) - 1L)
  scores_scaled <- pmax(pmin((x - score_min) / score_scale, 1), 0)
  knots <- .cc_choose_monotone_spline_knots(
    scores_scaled,
    max_internal_knots = as.integer(max_internal_knots),
    degree = derivative_degree
  )
  basis <- .cc_integrated_bspline_design(scores_scaled, knots = knots, degree = derivative_degree)
  n_basis <- ncol(basis)
  objective <- function(par) {
    intercept <- par[[1]]
    coef <- par[-1L]
    pred <- as.numeric(intercept + basis %*% coef)
    smoothness <- if (length(coef) < 3L) sum(coef^2) else sum(diff(coef, differences = 2L)^2)
    sum(weights * (y - pred)^2) + as.numeric(penalty) * smoothness
  }
  start_intercept <- .cc_weighted_mean(y, weights)
  start <- c(start_intercept, rep(0.1, n_basis))
  fit <- try(
    stats::optim(
      par = start,
      fn = objective,
      method = "L-BFGS-B",
      lower = c(-Inf, rep(0, n_basis))
    ),
    silent = TRUE
  )
  if (inherits(fit, "try-error") || is.null(fit$par) || any(!is.finite(fit$par))) {
    return(.cc_fit_linear(x, y, weights))
  }
  structure(
    list(
      kind = "monotone_spline",
      intercept = unname(fit$par[[1]]),
      coef = unname(fit$par[-1L]),
      knots = knots,
      derivative_degree = derivative_degree,
      score_min = score_min,
      score_scale = score_scale,
      y_min = y_min,
      y_max = y_max,
      metadata = list(
        max_internal_knots = as.integer(max_internal_knots),
        basis_degree = as.integer(basis_degree),
        penalty = as.numeric(penalty)
      )
    ),
    class = "cc_monotone_spline_model"
  )
}

.cc_predict_monotone_spline <- function(model, x) {
  values <- as.numeric(x)
  if (isTRUE(all.equal(model$score_scale, 0))) {
    return(rep(model$intercept, length(values)))
  }
  scaled <- pmax(pmin((values - model$score_min) / model$score_scale, 1), 0)
  prediction <- .cc_evaluate_monotone_spline(
    scores_scaled = scaled,
    knots = model$knots,
    coef = model$coef,
    degree = model$derivative_degree,
    intercept = model$intercept
  )
  pmin(pmax(prediction, model$y_min), model$y_max)
}

.cc_fit_backend <- function(method, x, y, weights, method_options = list()) {
  method <- .cc_validate_method(method)
  .cc_validate_min_unique_scores(x, method)
  if (method == "isotonic") {
    return(
      .cc_fit_isotonic(
        x = x,
        y = y,
        weights = weights,
        max_depth = if (!is.null(method_options$max_depth)) method_options$max_depth else 20L,
        min_child_samples = if (!is.null(method_options$min_child_samples)) method_options$min_child_samples else 10L,
        learning_rate = if (!is.null(method_options$learning_rate)) method_options$learning_rate else 1,
        n_estimators = if (!is.null(method_options$n_estimators)) method_options$n_estimators else 1L
      )
    )
  }
  if (method == "monotone_spline") {
    return(
      .cc_fit_monotone_spline(
        x = x,
        y = y,
        weights = weights,
        max_internal_knots = if (!is.null(method_options$max_internal_knots)) method_options$max_internal_knots else .cc_monotone_spline_max_internal_knots,
        basis_degree = if (!is.null(method_options$basis_degree)) method_options$basis_degree else .cc_monotone_spline_derivative_degree + 1L,
        penalty = if (!is.null(method_options$penalty)) method_options$penalty else .cc_monotone_spline_penalty
      )
    )
  }
  if (method == "linear") {
    return(.cc_fit_linear(x, y, weights))
  }
  if (method == "histogram") {
    n_bins <- if (!is.null(method_options$n_bins)) as.integer(method_options$n_bins) else 10L
    return(.cc_fit_histogram(x, y, weights, n_bins = n_bins))
  }
  stop("`method` must be one of 'isotonic', 'monotone_spline', 'linear', or 'histogram'.", call. = FALSE)
}

.cc_predict_backend <- function(model, x) {
  values <- as.numeric(x)
  if (inherits(model, "cc_histogram_model")) {
    return(.cc_predict_histogram(model, values))
  }
  if (inherits(model, "cc_linear_model")) {
    return(.cc_predict_linear(model, values))
  }
  if (inherits(model, "cc_lightgbm_isotonic_model")) {
    return(.cc_predict_lightgbm_isotonic(model, values))
  }
  if (inherits(model, "cc_monotone_spline_model")) {
    return(.cc_predict_monotone_spline(model, values))
  }
  stop("Unknown calibration backend.", call. = FALSE)
}

.cc_mapping_frame <- function(model) {
  if (inherits(model, "cc_histogram_model")) {
    return(
      data.frame(
        lower = model$lower_bounds,
        upper = model$upper_bounds,
        value = model$values
      )
    )
  }
  if (inherits(model, "cc_linear_model")) {
    return(data.frame(intercept = model$intercept, slope = model$slope))
  }
  if (inherits(model, "cc_lightgbm_isotonic_model")) {
    grid <- .cc_mapping_grid(c(model$score_min, model$score_max))
    return(data.frame(prediction = grid, calibrated = .cc_predict_lightgbm_isotonic(model, grid)))
  }
  grid <- .cc_mapping_grid(c(model$score_min, model$score_min + model$score_scale))
  data.frame(prediction = grid, calibrated = .cc_predict_monotone_spline(model, grid))
}
