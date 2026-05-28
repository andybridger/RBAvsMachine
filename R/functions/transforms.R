# =============================================================================
# Stationarity Transformation Functions
# =============================================================================
# FRED-MD style transformation codes (McCracken & Ng, 2016)

apply_tcode <- function(x, tcode) {
  n <- length(x)
  switch(as.character(tcode),
    "1" = x,
    "2" = c(NA, diff(x)),
    "3" = c(NA, NA, diff(x, differences = 2)),
    "4" = log(pmax(x, 1e-8)),
    "5" = c(NA, diff(log(pmax(x, 1e-8)))),
    "6" = c(NA, NA, diff(log(pmax(x, 1e-8)), differences = 2)),
    "7" = {
      pct <- x / dplyr::lag(x) - 1
      c(NA, diff(pct))
    },
    x
  )
}

tcode_label <- function(tcode) {
  labels <- c(
    "1" = "Level",
    "2" = "First difference",
    "3" = "Second difference",
    "4" = "Log level",
    "5" = "Log first difference",
    "6" = "Log second difference",
    "7" = "Percent change difference"
  )
  labels[as.character(tcode)]
}

select_tcode <- function(x) {
  x_clean <- x[!is.na(x)]
  if (length(x_clean) < 20) return(1L)
  if (any(x_clean <= 0)) {
    adf_level <- tryCatch(tseries::adf.test(x_clean)$p.value, error = function(e) 1)
    if (adf_level < 0.05) return(1L)
    adf_d1 <- tryCatch(tseries::adf.test(diff(x_clean))$p.value, error = function(e) 1)
    if (adf_d1 < 0.05) return(2L)
    return(3L)
  }
  adf_level <- tryCatch(tseries::adf.test(x_clean)$p.value, error = function(e) 1)
  if (adf_level < 0.05) return(1L)
  x_log <- log(pmax(x_clean, 1e-8))
  adf_dlog <- tryCatch(tseries::adf.test(diff(x_log))$p.value, error = function(e) 1)
  if (adf_dlog < 0.05) return(5L)
  adf_d2log <- tryCatch(tseries::adf.test(diff(x_log, differences = 2))$p.value, error = function(e) 1)
  if (adf_d2log < 0.05) return(6L)
  return(5L)
}
