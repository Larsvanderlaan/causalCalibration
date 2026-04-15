.cc_collapse_sorted_points <- function(x, y, weights) {
  ord <- order(x, y)
  x <- x[ord]
  y <- y[ord]
  weights <- weights[ord]
  keep <- weights > 0
  x <- x[keep]
  y <- y[keep]
  weights <- weights[keep]
  if (length(x) == 0) {
    stop("At least one observation must have positive weight.", call. = FALSE)
  }
  unique_x <- numeric()
  unique_y <- numeric()
  unique_w <- numeric()
  for (index in seq_along(x)) {
    if (length(unique_x) > 0 && x[index] == unique_x[length(unique_x)]) {
      updated_weight <- unique_w[length(unique_w)] + weights[index]
      unique_y[length(unique_y)] <- (
        unique_y[length(unique_y)] * unique_w[length(unique_w)] + y[index] * weights[index]
      ) / updated_weight
      unique_w[length(unique_w)] <- updated_weight
    } else {
      unique_x <- c(unique_x, x[index])
      unique_y <- c(unique_y, y[index])
      unique_w <- c(unique_w, weights[index])
    }
  }
  list(x = unique_x, y = unique_y, weights = unique_w)
}

.cc_fit_isotonic_blocks <- function(x, y, weights) {
  collapsed <- .cc_collapse_sorted_points(x, y, weights)
  blocks <- list()
  for (index in seq_along(collapsed$x)) {
    blocks[[length(blocks) + 1L]] <- list(
      lower = collapsed$x[index],
      upper = collapsed$x[index],
      weight = collapsed$weights[index],
      weighted_x = collapsed$x[index] * collapsed$weights[index],
      weighted_y = collapsed$y[index] * collapsed$weights[index],
      value = collapsed$y[index]
    )
    while (length(blocks) >= 2 && blocks[[length(blocks) - 1L]]$value > blocks[[length(blocks)]]$value) {
      right <- blocks[[length(blocks)]]
      left <- blocks[[length(blocks) - 1L]]
      blocks <- blocks[seq_len(length(blocks) - 2L)]
      total_weight <- left$weight + right$weight
      weighted_x <- left$weighted_x + right$weighted_x
      weighted_y <- left$weighted_y + right$weighted_y
      blocks[[length(blocks) + 1L]] <- list(
        lower = left$lower,
        upper = right$upper,
        weight = total_weight,
        weighted_x = weighted_x,
        weighted_y = weighted_y,
        value = weighted_y / total_weight
      )
    }
  }
  knots_x <- vapply(blocks, function(block) block$weighted_x / block$weight, numeric(1))
  knots_y <- vapply(blocks, function(block) block$value, numeric(1))
  list(blocks = blocks, knots_x = knots_x, knots_y = knots_y)
}

.cc_endpoint_slope <- function(h0, h1, d0, d1) {
  slope <- ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
  if (slope * d0 <= 0) {
    return(0)
  }
  if (d0 * d1 < 0 && abs(slope) > abs(3 * d0)) {
    return(3 * d0)
  }
  slope
}

.cc_monotone_cubic_slopes <- function(knots_x, knots_y) {
  n_knots <- length(knots_x)
  if (n_knots == 1) {
    return(0)
  }
  h <- diff(knots_x)
  delta <- diff(knots_y) / h
  if (n_knots == 2) {
    return(c(delta[1], delta[1]))
  }
  slopes <- rep(0, n_knots)
  slopes[1] <- .cc_endpoint_slope(h[1], h[2], delta[1], delta[2])
  slopes[n_knots] <- .cc_endpoint_slope(h[length(h)], h[length(h) - 1], delta[length(delta)], delta[length(delta) - 1])
  for (index in 2:(n_knots - 1)) {
    if (delta[index - 1] == 0 || delta[index] == 0 || delta[index - 1] * delta[index] < 0) {
      slopes[index] <- 0
    } else {
      w1 <- 2 * h[index] + h[index - 1]
      w2 <- h[index] + 2 * h[index - 1]
      slopes[index] <- (w1 + w2) / ((w1 / delta[index - 1]) + (w2 / delta[index]))
    }
  }
  slopes
}

.cc_predict_step <- function(model, x) {
  vapply(
    x,
    function(value) {
      idx <- findInterval(value, model$lower_bounds)
      idx <- max(1L, min(idx, length(model$values)))
      model$values[idx]
    },
    numeric(1)
  )
}

.cc_predict_smooth <- function(model, x) {
  if (length(model$knots_x) == 1) {
    return(rep(model$knots_y[1], length(x)))
  }
  vapply(
    x,
    function(value) {
      if (value <= model$knots_x[1]) {
        return(model$knots_y[1])
      }
      if (value >= model$knots_x[length(model$knots_x)]) {
        return(model$knots_y[length(model$knots_y)])
      }
      idx <- findInterval(value, model$knots_x)
      idx <- max(1L, min(idx, length(model$knots_x) - 1L))
      x0 <- model$knots_x[idx]
      x1 <- model$knots_x[idx + 1L]
      y0 <- model$knots_y[idx]
      y1 <- model$knots_y[idx + 1L]
      m0 <- model$slopes[idx]
      m1 <- model$slopes[idx + 1L]
      h <- x1 - x0
      t <- (value - x0) / h
      h00 <- 2 * t^3 - 3 * t^2 + 1
      h10 <- t^3 - 2 * t^2 + t
      h01 <- -2 * t^3 + 3 * t^2
      h11 <- t^3 - t^2
      h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    },
    numeric(1)
  )
}

.cc_predict_linear <- function(model, x) {
  model$intercept + model$slope * x
}

.cc_fit_isotonic <- function(x, y, weights) {
  fit <- .cc_fit_isotonic_blocks(x, y, weights)
  blocks <- fit$blocks
  structure(
    list(
      kind = "isotonic",
      lower_bounds = vapply(blocks, function(block) block$lower, numeric(1)),
      upper_bounds = vapply(blocks, function(block) block$upper, numeric(1)),
      values = vapply(blocks, function(block) block$value, numeric(1))
    ),
    class = "cc_step_model"
  )
}

.cc_fit_smooth_isotonic <- function(x, y, weights) {
  fit <- .cc_fit_isotonic_blocks(x, y, weights)
  structure(
    list(
      kind = "smooth_isotonic",
      knots_x = fit$knots_x,
      knots_y = fit$knots_y,
      slopes = .cc_monotone_cubic_slopes(fit$knots_x, fit$knots_y)
    ),
    class = "cc_smooth_model"
  )
}

.cc_fit_linear <- function(x, y, weights) {
  x_mean <- .cc_weighted_mean(x, weights)
  y_mean <- .cc_weighted_mean(y, weights)
  denominator <- sum(weights * (x - x_mean)^2)
  if (denominator == 0) {
    return(structure(list(kind = "linear", intercept = y_mean, slope = 0), class = "cc_linear_model"))
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
    cumulative <- cumulative + weights[index]
    while (target_index <= length(targets) && cumulative >= targets[target_index]) {
      if (length(breaks) == 0 || x[index] > breaks[length(breaks)]) {
        breaks <- c(breaks, x[index])
      }
      target_index <- target_index + 1L
    }
  }
  breaks
}

.cc_fit_histogram <- function(x, y, weights, n_bins = 10L) {
  if (n_bins < 1) {
    stop("`n_bins` must be at least 1.", call. = FALSE)
  }
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]
  weights <- weights[ord]
  breaks <- .cc_weighted_quantile_breaks(x, weights, n_bins)
  lower_bounds <- c(x[1], breaks[breaks > x[1]])
  bin_ids <- findInterval(x, lower_bounds)
  bin_ids[bin_ids < 1L] <- 1L
  values <- numeric(length(lower_bounds))
  upper_bounds <- numeric(length(lower_bounds))
  overall_mean <- .cc_weighted_mean(y, weights)
  for (index in seq_along(lower_bounds)) {
    in_bin <- which(bin_ids == index)
    if (length(in_bin) == 0) {
      values[index] <- if (index == 1L) overall_mean else values[index - 1L]
      upper_bounds[index] <- lower_bounds[index]
    } else {
      values[index] <- .cc_weighted_mean(y[in_bin], weights[in_bin])
      upper_bounds[index] <- max(x[in_bin])
    }
  }
  structure(
    list(
      kind = "histogram",
      lower_bounds = lower_bounds,
      upper_bounds = upper_bounds,
      values = values
    ),
    class = "cc_step_model"
  )
}

.cc_fit_backend <- function(method, x, y, weights, method_options = list()) {
  if (method == "isotonic") {
    return(.cc_fit_isotonic(x, y, weights))
  }
  if (method == "smooth_isotonic") {
    return(.cc_fit_smooth_isotonic(x, y, weights))
  }
  if (method == "linear") {
    return(.cc_fit_linear(x, y, weights))
  }
  if (method == "histogram") {
    n_bins <- if (!is.null(method_options$n_bins)) as.integer(method_options$n_bins) else 10L
    return(.cc_fit_histogram(x, y, weights, n_bins = n_bins))
  }
  stop("`method` must be one of 'isotonic', 'smooth_isotonic', 'linear', or 'histogram'.", call. = FALSE)
}

.cc_predict_backend <- function(model, x) {
  x <- as.numeric(x)
  if (inherits(model, "cc_step_model")) {
    return(.cc_predict_step(model, x))
  }
  if (inherits(model, "cc_smooth_model")) {
    return(.cc_predict_smooth(model, x))
  }
  if (inherits(model, "cc_linear_model")) {
    return(.cc_predict_linear(model, x))
  }
  stop("Unknown calibration backend.", call. = FALSE)
}

.cc_mapping_frame <- function(model) {
  if (inherits(model, "cc_step_model")) {
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
  data.frame(x = model$knots_x, y = model$knots_y, slope = model$slopes)
}
