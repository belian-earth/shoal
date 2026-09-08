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
#' Rust and never held in memory, so the cost is `(n m + n^2 + m^2) d / 2`
#' flops and the memory is the two inputs. For unit-normalised rows the
#' Euclidean distance is a monotone function of cosine distance, so the
#' statistic can be used on embeddings that are compared by cosine.
#'
#' @param x,y Numeric matrices or data frames with the same number of
#'   columns. Data frames are coerced to matrices using their numeric
#'   columns. Rows with missing or non-finite values are an error.
#' @param terms Also return the three mean distances. Default `FALSE`.
#'
#' @returns The energy distance, a single non-negative number; with
#'   `terms = TRUE`, a list with `energy`, `cross` (mean cross-pair
#'   distance), `self_x` and `self_y` (mean within-sample distances).
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
#' @export
shoal_energy <- function(x, y, terms = FALSE) {
  x <- check_numeric_matrix(x, na_action = "error")
  y <- check_numeric_matrix(y, na_action = "error")
  if (ncol(x) != ncol(y)) {
    cli::cli_abort(c(
      "{.arg x} and {.arg y} must have the same number of columns.",
      "i" = "{.arg x} has {ncol(x)} and {.arg y} has {ncol(y)}."
    ))
  }
  res <- rust_energy(x, y)
  e <- max(0, 2 * res$cross - res$self_x - res$self_y)
  if (isTRUE(terms)) {
    list(energy = e, cross = res$cross, self_x = res$self_x, self_y = res$self_y)
  } else {
    e
  }
}
