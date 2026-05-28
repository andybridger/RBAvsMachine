# =============================================================================
# Forecast Evaluation Helper Functions
# =============================================================================

calc_rmsfe <- function(actual, forecast) {
  e <- actual - forecast
  sqrt(mean(e^2, na.rm = TRUE))
}

calc_mafe <- function(actual, forecast) {
  mean(abs(actual - forecast), na.rm = TRUE)
}

calc_bias <- function(actual, forecast) {
  mean(actual - forecast, na.rm = TRUE)
}

calc_all_metrics <- function(actual, forecast) {
  tibble(
    rmsfe = calc_rmsfe(actual, forecast),
    mafe  = calc_mafe(actual, forecast),
    bias  = calc_bias(actual, forecast),
    n     = sum(!is.na(actual) & !is.na(forecast))
  )
}

dm_test <- function(e1, e2, h = 1, alternative = "two.sided") {
  d <- e1^2 - e2^2
  n <- length(d)
  if (n < 5) return(list(statistic = NA, p.value = NA))
  d_bar <- mean(d)
  if (h > 1) {
    gamma <- sapply(0:(h - 1), function(k) {
      cov(d[1:(n - k)], d[(1 + k):n])
    })
    V <- (gamma[1] + 2 * sum(gamma[-1])) / n
  } else {
    V <- var(d) / n
  }
  if (V <= 0) return(list(statistic = NA, p.value = NA))
  stat <- d_bar / sqrt(V)
  p <- switch(alternative,
    "two.sided" = 2 * pnorm(-abs(stat)),
    "less"      = pnorm(stat),
    "greater"   = pnorm(-stat)
  )
  list(statistic = stat, p.value = p)
}

relative_rmsfe <- function(rmsfe_model, rmsfe_benchmark) {
  rmsfe_model / rmsfe_benchmark
}
