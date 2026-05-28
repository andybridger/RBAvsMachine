# =============================================================================
# Data Pull Helper Functions
# =============================================================================
# This file contains helper functions for pulling data from various sources:
# - ABS (Australian Bureau of Statistics)
# - RBA (Reserve Bank of Australia)
# - Yahoo Finance
# - FRED (Federal Reserve Economic Data)
# =============================================================================

# -----------------------------------------------------------------------------
# ABS Data Functions
# -----------------------------------------------------------------------------

#' Pull ABS data in chunks to avoid API limits
#'
#' @param series_ids Character vector of ABS series IDs
#' @param chunk_size Number of series to pull per API call (default: 100)
#' @return Data frame with all ABS data
pull_abs_data <- function(series_ids, chunk_size = 100) {
  # Split into chunks to avoid API limits
  chunks <- split(series_ids, ceiling(seq_along(series_ids) / chunk_size))

  # Pull each chunk and combine
  results <- purrr::map(chunks, function(chunk) {
    tryCatch(
      readabs::read_abs(series_id = chunk),
      error = function(e) {
        warning(paste("Error pulling ABS chunk:", e$message))
        return(NULL)
      }
    )
  })

  # Combine non-null results
  dplyr::bind_rows(purrr::compact(results))
}

#' Standardize ABS data to common format
#'
#' @param df ABS data frame from read_abs
#' @return Standardized data frame
standardize_abs <- function(df) {
  df |>
    dplyr::mutate(package = "readabs") |>
    dplyr::select(package, date, series, series_id, series_type, frequency, value)
}

# -----------------------------------------------------------------------------
# RBA Data Functions
# -----------------------------------------------------------------------------

#' Pull RBA data with error handling
#'
#' @param series_ids Character vector of RBA series IDs
#' @return Data frame with all RBA data
pull_rba_data <- function(series_ids) {
  results <- purrr::map(series_ids, function(id) {
    tryCatch(
      readrba::read_rba(series_id = id),
      error = function(e) {
        warning(paste("Error pulling RBA series", id, ":", e$message))
        return(NULL)
      }
    )
  })

  # Combine non-null results
  dplyr::bind_rows(purrr::compact(results))
}

#' Standardize RBA data to common format
#'
#' @param df RBA data frame from read_rba
#' @return Standardized data frame
standardize_rba <- function(df) {
  df |>
    dplyr::mutate(package = "readrba") |>
    dplyr::select(package, date, series, series_id, series_type, frequency, value)
}

# -----------------------------------------------------------------------------
# Yahoo Finance Functions
# -----------------------------------------------------------------------------
#' Pull Yahoo Finance data for multiple tickers
#'
#' @param tickers_df Data frame with columns: ticker, series_name
#' @param start_date Start date for data pull (default: "1990-01-01")
#' @param end_date End date for data pull (default: current date)
#' @return Data frame with all Yahoo Finance data
pull_yahoo_data <- function(tickers_df, start_date = "1990-01-01", end_date = NULL) {
  if (is.null(end_date)) {
    end_date <- as.character(Sys.Date())
  }

  results <- purrr::map(seq_len(nrow(tickers_df)), function(i) {
    tryCatch({
      ticker_obj <- yahoofinancer::Ticker$new(tickers_df$ticker[i])
      df <- ticker_obj$get_history(start = start_date, end = end_date, interval = "1mo")

      df |>
        dplyr::mutate(
          series_id = tickers_df$ticker[i],
          series = tickers_df$series_name[i]
        )
    }, error = function(e) {
      warning(paste("Error pulling Yahoo ticker", tickers_df$ticker[i], ":", e$message))
      return(NULL)
    })
  })

  # Combine and add metadata
dplyr::bind_rows(purrr::compact(results)) |>
    dplyr::mutate(
      package = "yahoofinancer",
      frequency = "Monthly",
      series_type = "Original"
    )
}

#' Standardize Yahoo Finance data to common format
#'
#' @param df Yahoo Finance data frame
#' @return Standardized data frame
standardize_yahoo <- function(df) {
  df |>
    dplyr::rename(value = close) |>
    dplyr::select(package, date, series, series_id, series_type, frequency, value)
}

# -----------------------------------------------------------------------------
# FRED Functions
# -----------------------------------------------------------------------------

#' Pull FRED data for multiple series
#'
#' @param series_df Data frame with columns: series_id, series_name
#' @param start_date Start date for data pull (default: "1990-01-01")
#' @param end_date End date for data pull (default: current date)
#' @return Data frame with all FRED data
pull_fred_data <- function(series_df, start_date = "1990-01-01", end_date = NULL) {
  if (is.null(end_date)) {
    end_date <- Sys.Date()
  }

  results <- purrr::map(seq_len(nrow(series_df)), function(i) {
    tryCatch({
      fredr::fredr(
        series_id = series_df$series_id[i],
        observation_start = as.Date(start_date),
        observation_end = as.Date(end_date),
        frequency = "m",
        units = "lin"
      ) |>
        dplyr::mutate(series = series_df$series_name[i])
    }, error = function(e) {
      warning(paste("Error pulling FRED series", series_df$series_id[i], ":", e$message))
      return(NULL)
    })
  })

  # Combine and add metadata
  dplyr::bind_rows(purrr::compact(results)) |>
    dplyr::mutate(
      package = "fredr",
      series_type = "Original",
      frequency = "Monthly"
    )
}

#' Standardize FRED data to common format
#'
#' @param df FRED data frame
#' @return Standardized data frame
standardize_fred <- function(df) {
  df |>
    dplyr::select(package, date, series, series_id, series_type, frequency, value)
}

# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------

#' Validate data pull by checking unique series count
#'
#' @param df Data frame to validate
#' @param expected_count Expected number of unique series
#' @param source_name Name of data source for logging
#' @return Invisible TRUE if valid, warning if count mismatch
validate_series_count <- function(df, expected_count, source_name) {
  actual_count <- dplyr::n_distinct(df$series_id)

  if (actual_count != expected_count) {
    warning(sprintf(
      "%s: Expected %d series, got %d (missing %d)",
      source_name,
      expected_count,
      actual_count,
      expected_count - actual_count
    ))
  } else {
    message(sprintf("%s: Successfully pulled %d series", source_name, actual_count))
  }

  invisible(actual_count == expected_count)
}

#' Get missing series IDs
#'
#' @param requested Character vector of requested series IDs
#' @param received Data frame with series_id column
#' @return Character vector of missing series IDs
get_missing_series <- function(requested, received) {
  received_ids <- unique(received$series_id)
  setdiff(requested, received_ids)
}
