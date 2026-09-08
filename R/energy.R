#' Two-Sample Energy Distance
#'
#' The energy distance of Szekely and Rizzo between the rows of `x` and the
#' rows of `y`, treated as two samples:
#'
#' \deqn{E(X, Y) = 2\,\mathrm{E}\|x - y\| - \mathrm{E}\|x - x'\| - \mathrm{E}\|y - y'\|}
#'
#' estimated by the mean Euclidean distance over all cross pairs and over
#' all distinct within-sample pairs. It is zero exactly when the two
#' distributions agree, positive otherwise, and needs no clustering, density
#' estimate or choice of bins, so it compares point clouds of any dimension
#' directly: two embeddings of the same region in different years, a sample
#' against a reference population, or a cluster against the rest.
#'
#' The three pairwise-distance blocks are reduced row by row in parallel in
#' Rust and never held in memory, so the cost is about
#' `(2 n m + n^2 + m^2) d / 2` flops and the memory is the two inputs. When
#' one sample is compared against many others its within-sample term is the
#' same every time: `shoal_energy_self()` computes that term alone, and
#' passing it as `self_x` or `self_y` skips recomputing it. For unit-normalised rows the
#' Euclidean distance is a monotone function of cosine distance, so the
#' statistic can be used on embeddings that are compared by cosine.
#'
#' @param x,y Numeric matrices or data frames with the same number of
#'   columns, one included. Data frames are coerced to matrices using their
#'   numeric columns. Rows with missing or non-finite values are an error.
#' @param terms Also return the three mean distances. Default `FALSE`.
#' @param self_x,self_y Optional precomputed within-sample terms from
#'   `shoal_energy_self()`, used in place of computing them.
#'
#' The cross term averages over all `n m` pairs, including the zero
#' distances between a row and its own copy when `x` and `y` share rows, so
#' a sample compared with itself comes out slightly below zero before the
#' floor at zero is applied.
#'
#' @returns `shoal_energy()`: the energy distance, a single non-negative
#'   number; with `terms = TRUE`, a list with `energy`, `cross` (mean
#'   cross-pair distance), `self_x` and `self_y` (mean within-sample
#'   distances). `shoal_energy_self()`: the mean distance over all distinct
#'   pairs of rows of `x`, `0` for a single row.
#'
#' @references Szekely GJ, Rizzo ML (2013). Energy statistics: A class of
#'   statistics based on distances. Journal of Statistical Planning and
#'   Inference 143(8), 1249-1272.
#'
#' @examples
#' x <- as.matrix(iris[iris$Species == "setosa", 1:4])
#' y <- as.matrix(iris[iris$Species == "versicolor", 1:4])
#' z <- as.matrix(iris[iris$Species == "virginica", 1:4])
#' shoal_energy(x, y)
#' shoal_energy(y, z)  # closer pair
#' shoal_energy(x, x, terms = TRUE)
#'
#' # One reference against several samples: compute its own term once.
#' self_x <- shoal_energy_self(x)
#' shoal_energy(x, y, self_x = self_x)
#' shoal_energy(x, z, self_x = self_x)
#'
#' @export
shoal_energy <- function(x, y, terms = FALSE, self_x = NULL, self_y = NULL) {
  x <- check_numeric_matrix(x, na_action = "error", min_cols = 1L)
  y <- check_numeric_matrix(y, na_action = "error", min_cols = 1L)
  if (ncol(x) != ncol(y)) {
    cli::cli_abort(c(
      "{.arg x} and {.arg y} must have the same number of columns.",
      "i" = "{.arg x} has {ncol(x)} and {.arg y} has {ncol(y)}."
    ))
  }
  check_self_term(self_x)
  check_self_term(self_y)
  res <- rust_energy(x, y, is.null(self_x), is.null(self_y))
  self_x <- self_x %||% res$self_x
  self_y <- self_y %||% res$self_y
  e <- max(0, 2 * res$cross - self_x - self_y)
  if (isTRUE(terms)) {
    list(energy = e, cross = res$cross, self_x = self_x, self_y = self_y)
  } else {
    e
  }
}

#' @rdname shoal_energy
#' @export
shoal_energy_self <- function(x) {
  x <- check_numeric_matrix(x, na_action = "error", min_cols = 1L)
  rust_energy_self(x)
}

check_self_term <- function(x, arg = rlang::caller_arg(x), call = rlang::caller_env()) {
  if (is.null(x)) {
    return(invisible(NULL))
  }
  if (!rlang::is_scalar_double(x) || is.na(x) || !is.finite(x) || x < 0) {
    cli::cli_abort("{.arg {arg}} must be a single non-negative number or NULL.", call = call)
  }
  invisible(x)
}
