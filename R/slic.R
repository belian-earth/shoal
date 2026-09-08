#' Graph SLIC Superpixels
#'
#' Partitions a point set into compact, feature-homogeneous patches by the
#' SLIC relaxation (Achanta et al. 2012) run on an arbitrary neighbour graph
#' rather than a raster. Starting from initial labels, every boundary point
#' is moved, all points at once, to the adjacent patch whose feature mean is
#' closest, with a spatial term that keeps patches compact:
#'
#' \deqn{\|x_i - m_p\|^2 + \lambda_p \, \|xy_i - c_p\|^2 / s_{nom}^2}
#'
#' where \eqn{m_p} and \eqn{c_p} are the patch's feature and coordinate means
#' at the start of the iteration. Because the adjacency and the seeding are
#' inputs, the same routine serves a raster, a hexagonal or pentagonal
#' discrete global grid, or any point cloud with a neighbour structure.
#'
#' # Compactness
#'
#' `lambda` weights the spatial term at the nominal radius `s_nom`, in the
#' units of `xy`: a point `s_nom` from its patch centre pays `lambda` in the
#' squared-feature-distance currency of `x`. The scale of that currency is
#' data dependent, so `init_stat` in the result gives the mean squared
#' feature distance of each initial patch's points to its mean, the local
#' variability statistic of `supercells::sc_tune_compactness()`. Expressing
#' `lambda` as a multiple of, say, its median carries a tuned setting to
#' other data. Balancing the two terms one-for-one (`lambda` equal to the
#' median) is a common default that over-regularises high-dimensional
#' embeddings; on those, values of a tenth of the median or less are what
#' track the data.
#'
#' `adaptive = "mean"` rescales the spatial term per patch to `alpha` times
#' that patch's own mean squared feature distance at the start of each
#' iteration, in the manner of ASLIC: the balance is then relative to local
#' contrast, so homogeneous and heterogeneous regions are regularised alike
#' in relative terms.
#'
#' # Post-processing
#'
#' After the iterations each patch is split into its connected components
#' over `nb`, and patches of fewer than `min_patch` points are absorbed into
#' the neighbouring patch they share the most edges with, until none remain.
#' Labels are then renumbered `1..n_clusters`.
#'
#' Every step is deterministic. Ties in the move decision go to the point's
#' own patch, then to the earlier column of `nb`; component ids follow first
#' appearance in row order.
#'
#' @param x A numeric matrix or data frame of features, one row per point.
#'   Data frames are coerced to a matrix using their numeric columns. Rows
#'   with missing or non-finite values are an error.
#' @param xy A numeric matrix with two columns of coordinates, one row per
#'   point, in the units `s_nom` is expressed in.
#' @param nb An integer matrix with one row per point whose columns hold the
#'   row indices of that point's neighbours, `NA` for an unused slot, as
#'   `shoal_knn()$id` or `dbscan::kNN()$id` give after distant neighbours are
#'   set to `NA`. Need not be symmetric.
#' @param init An integer vector of initial patch labels, one per point.
#'   Any labelling works; a regular tiling at the intended patch scale is
#'   the usual choice.
#' @param lambda Spatial weight at `s_nom`. Ignored when `adaptive` is not
#'   `"none"`.
#' @param s_nom Nominal radius, in the units of `xy`. Default `1`.
#' @param n_iter Iterations. Default `25L`. The `moves` component shows
#'   whether that was enough: the count typically decays geometrically.
#' @param adaptive `"none"` (default) for one global `lambda`, or `"mean"`
#'   for per-patch weights of `alpha` times the patch's mean squared
#'   feature distance.
#' @param alpha Multiplier for `adaptive = "mean"`.
#' @param min_patch Smallest patch kept after relaxation, in points. Default
#'   `1L` keeps every component.
#' @param tol Stop early once fewer than this share of points move in an
#'   iteration. Default `0` runs all `n_iter`.
#'
#' @returns An object of class `c("shoal_slic", "shoal_clustering")`: a list
#'   with components `cluster` (integer patch labels), `n_clusters`,
#'   `n_noise` (always `0`), `data`, `algorithm`, `params`, `moves` (points
#'   moved in each iteration run), `init_stat` (per initial label, mean
#'   squared feature distance to its mean, indexed by label) and
#'   `lambda_median` (median per-patch weight in each iteration run).
#'
#' @references Achanta R, Shaji A, Smith K, Lucchi A, Fua P, Süsstrunk S
#'   (2012). SLIC superpixels compared to state-of-the-art superpixel
#'   methods. IEEE Transactions on Pattern Analysis and Machine Intelligence
#'   34(11), 2274-2282.
#'
#' @examples
#' # A 20 x 20 grid with two feature regions and a 4-neighbour graph.
#' side <- 20L
#' g <- expand.grid(c = seq_len(side), r = seq_len(side))
#' x <- cbind(as.numeric(g$c > side / 2), 0.1 * g$r / side) + rnorm(2 * side^2, sd = 0.02)
#' xy <- cbind(g$c, g$r) * 10
#' nn <- shoal_knn(xy, k = 4L)
#' nb <- nn$id
#' nb[nn$dist > 10] <- NA
#' init <- as.integer(factor(paste((g$r - 1) %/% 4, (g$c - 1) %/% 4)))
#' sp <- shoal_slic(x, xy, nb, init, lambda = 0.01, s_nom = 20, n_iter = 10L)
#' sp
#' table(sp$cluster, x[, 1] > 0.5)[1:5, ]
#'
#' @export
shoal_slic <- function(x, xy, nb, init, lambda = 1, s_nom = 1, n_iter = 25L,
                       adaptive = c("none", "mean"), alpha = NULL,
                       min_patch = 1L, tol = 0) {
  x <- check_numeric_matrix(x, na_action = "error")
  n <- nrow(x)
  xy <- check_numeric_matrix(xy, na_action = "error")
  if (nrow(xy) != n || ncol(xy) != 2L) {
    cli::cli_abort("{.arg xy} must be a numeric matrix with {n} rows and 2 columns.")
  }
  nb <- check_neighbour_matrix(nb, n)
  if (!rlang::is_integerish(init) || length(init) != n || anyNA(init)) {
    cli::cli_abort("{.arg init} must be an integer vector of length {n} without missing values.")
  }
  init <- as.integer(factor(init))
  adaptive <- rlang::arg_match(adaptive)
  if (identical(adaptive, "none")) {
    check_positive_number(lambda)
    alpha <- 0
  } else {
    if (is.null(alpha)) {
      cli::cli_abort("{.arg alpha} is required for {.code adaptive = \"{adaptive}\"}.")
    }
    check_positive_number(alpha)
  }
  check_positive_number(s_nom)
  check_positive_integer(n_iter)
  check_positive_integer(min_patch)
  if (!rlang::is_scalar_double(tol) && !rlang::is_scalar_integer(tol) || is.na(tol) || tol < 0 || tol >= 1) {
    cli::cli_abort("{.arg tol} must be a number in [0, 1).")
  }

  res <- rust_slic(
    x, xy, nb, init, as.double(lambda), as.double(s_nom), as.integer(n_iter),
    adaptive, as.double(alpha), as.integer(min_patch), as.double(tol)
  )

  new_clustering(
    cluster = res$labels,
    data = x,
    algorithm = "Graph SLIC",
    subclass = "shoal_slic",
    params = list(
      lambda = if (identical(adaptive, "none")) lambda else NA_real_,
      s_nom = s_nom,
      n_iter = as.integer(n_iter),
      adaptive = adaptive,
      alpha = if (identical(adaptive, "none")) NA_real_ else alpha,
      min_patch = as.integer(min_patch)
    ),
    moves = as.integer(res$moves),
    init_stat = res$init_stat,
    lambda_median = res$lambda_median
  )
}

#' @rdname print.shoal
#' @export
print.shoal_slic <- function(x, ...) {
  NextMethod()
  m <- x$moves
  cli::cli_text(
    "Iterations run: {length(m)}; points moved: first {m[1]}, last {m[length(m)]}"
  )
  cli::cli_text(
    "Initial-patch variability (median): {signif(stats::median(x$init_stat), 4)}"
  )
  invisible(x)
}

check_neighbour_matrix <- function(nb, n, arg = rlang::caller_arg(nb), call = rlang::caller_env()) {
  if (!is.matrix(nb) || !rlang::is_integerish(nb)) {
    cli::cli_abort("{.arg {arg}} must be an integer matrix of neighbour row indices.", call = call)
  }
  if (nrow(nb) != n) {
    cli::cli_abort("{.arg {arg}} must have one row per point ({n}).", call = call)
  }
  storage.mode(nb) <- "integer"
  bad <- !is.na(nb) & (nb < 1L | nb > n)
  if (any(bad)) {
    cli::cli_abort("{.arg {arg}} has {sum(bad)} entr{?y/ies} outside 1..{n}; use NA for an unused slot.", call = call)
  }
  nb
}
