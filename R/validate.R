# `min_cols` is 2 for the clustering algorithms, where a one-column input is
# almost always a mistake, and 1 where a single feature is a legitimate use
# (a one-band raster for SLIC, a univariate energy distance).
check_numeric_matrix <- function(x, na_action = c("drop", "error"),
                                 min_cols = 2L,
                                 arg = rlang::caller_arg(x),
                                 call = rlang::caller_env()) {
  na_action <- rlang::arg_match(na_action)

  if (is.data.frame(x)) {
    numeric_cols <- vapply(x, is.numeric, logical(1L))
    if (sum(numeric_cols) < min_cols) {
      cli::cli_abort(
        "{.arg {arg}} data frame must have at least {min_cols} numeric column{?s}.",
        call = call
      )
    }
    x <- as.matrix(x[, numeric_cols, drop = FALSE])
  }
  if (!is.matrix(x) || !is.numeric(x)) {
    cli::cli_abort("{.arg {arg}} must be a numeric matrix or data frame.", call = call)
  }
  if (nrow(x) < 1L) {
    cli::cli_abort("{.arg {arg}} must have at least 1 row.", call = call)
  }
  if (ncol(x) < min_cols) {
    cli::cli_abort("{.arg {arg}} must have at least {min_cols} column{?s}.", call = call)
  }

  # extendr expects a double matrix; an integer one (0/1 presence data, say)
  # would otherwise fail to convert.
  if (!is.double(x)) {
    storage.mode(x) <- "double"
  }

  # Non-finite values (NA, NaN, Inf) all poison distance computations, and
  # complete.cases() only catches the first two, so screen on finiteness.
  finite <- is.finite(x)
  if (!all(finite)) {
    incomplete <- rowSums(!finite) > 0L
    n_drop <- sum(incomplete)
    if (identical(na_action, "error")) {
      cli::cli_abort(
        "{.arg {arg}} contains {n_drop} row{?s} with missing or non-finite values.",
        call = call
      )
    }
    cli::cli_warn(
      c("Removed {n_drop} row{?s} containing missing or non-finite values.",
        "i" = "{nrow(x) - n_drop} complete row{?s} remaining."),
      call = call
    )
    x <- x[!incomplete, , drop = FALSE]
    if (nrow(x) < 1L) {
      cli::cli_abort(
        "No complete rows remaining after removing missing or non-finite values.",
        call = call
      )
    }
  }
  x
}

check_positive_number <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (!rlang::is_scalar_double(x) && !rlang::is_scalar_integer(x)) {
    cli::cli_abort("{.arg {arg}} must be a single number.", call = call)
  }
  if (is.na(x) || !is.finite(x) || x <= 0) {
    cli::cli_abort("{.arg {arg}} must be a positive, finite number.", call = call)
  }
  invisible(x)
}

check_positive_integer <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (!rlang::is_scalar_integerish(x)) {
    cli::cli_abort("{.arg {arg}} must be a single integer.", call = call)
  }
  if (is.na(x) || x < 1L) {
    cli::cli_abort("{.arg {arg}} must be a positive integer.", call = call)
  }
  invisible(x)
}

check_count <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (!rlang::is_scalar_integerish(x)) {
    cli::cli_abort("{.arg {arg}} must be a single integer.", call = call)
  }
  if (is.na(x) || x < 0L) {
    cli::cli_abort("{.arg {arg}} must be a non-negative integer.", call = call)
  }
  invisible(x)
}

#' Check newdata against the columns a model was fitted on
#'
#' Aborts on a column-count mismatch. A pure name or order difference only
#' warns: unnamed matrices are legitimate, but silently predicting on reordered
#' columns would be a wrong answer, so it should not pass without comment.
#'
#' @noRd
check_newdata_columns <- function(newdata, fitted, call = rlang::caller_env()) {
  if (ncol(newdata) != ncol(fitted)) {
    cli::cli_abort(
      "{.arg newdata} has {ncol(newdata)} column{?s}, but the model was fitted on {ncol(fitted)}.",
      call = call
    )
  }

  new_names <- colnames(newdata)
  fit_names <- colnames(fitted)
  if (!is.null(new_names) && !is.null(fit_names) && !identical(new_names, fit_names)) {
    cli::cli_warn(
      c("Column names of {.arg newdata} differ from those the model was fitted on.",
        "i" = "Columns are matched by position, not name."),
      call = call
    )
  }

  invisible(newdata)
}
