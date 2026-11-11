#' e-means Clustering
#'
#' k-means fitted by e-means (Bessa et al. 2026), via an in-tree Rust port
#' of the reference implementation in
#' \href{https://github.com/facebookresearch/faiss/tree/main/contrib/e_means}{faiss}
#' (Meta FAIR). Lloyd's algorithm, behind [shoal_kmeans()], recomputes every
#' centroid from all the data at each iteration. e-means visits the data in
#' random mini-batches instead, moving each centroid towards its batch mean
#' by an exponential moving average, with the batch size and step size set
#' by scaling laws the authors fitted on the number of rows, clusters and
#' passes. It minimises the same within-cluster sum of squares, and reaches
#' the loss of a converged k-means in a handful of passes over the data,
#' which makes large `k` on large data practical.
#'
#' # When to use it
#'
#' e-means is for scale: many rows and many clusters, such as codebooks,
#' vector quantisation or a fine partition of an embedding collection. The
#' scaling laws were fitted on \eqn{10^5} to \eqn{10^8} rows and 256 to
#' 65,536 clusters. Outside that range e-means still runs, but its settings
#' are not tuned there. For a few clusters, prefer [shoal_kmeans()]: its
#' k-means++ start avoids the poor local minima a random start can settle
#' in when `k` is small, and on small data Lloyd's algorithm converges in
#' moments anyway.
#'
#' The cost is fixed by `epochs`, one pass over the data per epoch plus one
#' for the final assignment, with no restarts and no convergence test.
#'
#' # Cosine mode
#'
#' `metric = "cosine"` is spherical k-means, for embedding vectors compared
#' by cosine similarity. Rows are scaled to unit length internally, each is
#' assigned to the centroid of largest cosine similarity, and the centroids
#' are unit vectors: the normalised mean directions of their clusters.
#' Rescaling rows therefore has no effect. A row of all zeros has no
#' direction and is an error. `inertia` is then the sum of squared distances
#' between the unit rows and their unit centroids, \eqn{\sum 2(1 - \cos\theta)}.
#'
#' # The training plan
#'
#' Unless given, the batch size and the initial step `alpha0` come from the
#' fitted laws, and the step is cosine-annealed to a tenth of `alpha0` over
#' the run. `batch_law = "quality"` uses batches about eight times smaller
#' than `"throughput"`, trading speed for a lower final loss; `"auto"`
#' chooses `"quality"` for data with at most 16 columns and `"throughput"`
#' above, as upstream does. Every five updates, clusters whose running mass
#' has fallen below `revive_threshold` times the mean are revived by
#' splitting a large one. The plan actually used is returned in `recipe`.
#'
#' # Assignments
#'
#' `cluster` is each row's nearest final centroid, from one exact pass after
#' training, so `cluster`, `centroids`, `inertia` and `sizes` describe the
#' same partition and agree with [predict()]. (The reference implementation
#' saves that pass by returning the labels from the last training epoch,
#' while the centroids were still moving.) A centroid can end the run with
#' no points; `sizes` then has a zero for it, and `n_clusters` counts only
#' the clusters with points.
#'
#' # Reproducibility
#'
#' As in [shoal_kmeans()], `seed` is a parameter rather than being taken
#' from R's RNG. The same seed, parameters and data give identical results
#' whatever the thread count. A seed does not reproduce a run of the
#' reference implementation, whose random streams differ; the port is
#' checked against it with the start and batch order fixed instead.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a
#'   matrix using their numeric columns (non-numeric columns are dropped).
#' @param k Number of clusters. Must be at least 1 and no more than
#'   `nrow(x)`.
#' @param epochs Number of passes over the data, the training budget.
#'   Fractional values train on a random fraction of the rows for one of
#'   the passes. Default `5`, upstream's.
#' @param metric `"euclidean"` (default) for k-means, or `"cosine"` for
#'   spherical k-means on directions; see the Cosine mode section.
#' @param batch_law Which fitted law sets the batch size when `batch_size`
#'   is not given: `"auto"` (default), `"throughput"` or `"quality"`.
#' @param batch_size Rows per update. `NULL` (default) uses the batch law.
#' @param alpha0 Initial step of the moving average, in \eqn{(0, 1]}. `NULL`
#'   (default) uses the step law at the batch size in use.
#' @param revive_threshold Revive clusters whose running mass falls below
#'   this fraction of the mean mass; `0` turns revival off. `NULL` (default)
#'   uses `0.1`, or `0` for data with at most two columns, where upstream
#'   found revival costs accuracy.
#' @param seed Non-negative whole-number seed for the start, the batch order
#'   and revival. Stored and passed as a double, so values beyond the
#'   integer range are safe. Default `1L`.
#'
#' @returns An object of class
#'   `c("shoal_emeans", "shoal_kmeans", "shoal_clustering")`: a list with
#'   components `cluster` (integer vector of cluster IDs), `n_clusters`,
#'   `n_noise` (always `0`), `data`, `algorithm`, `params`, `metric`,
#'   `centroids` (a `k x ncol(x)` matrix), `inertia`, `sizes`,
#'   `epoch_objective` (per epoch, the sum of squared distances of the rows
#'   visited to the centroids they were assigned against, which move during
#'   the epoch: a training trace, not the final error), `n_splits` (clusters
#'   revived) and `recipe`, the plan used: `batch_law`, `batch_size`,
#'   `alpha0`, `epoch_rows` (rows visited per epoch), `total_steps` and
#'   `revive_threshold`.
#'
#' @references
#' Bessa, S., Mazaré, P.-E., Szilvasy, G., Touvron, H., Picard, D., Douze, M.
#' and Jégou, H. (2026). *e-means: outperforming k-means (in one-tenth the
#' epochs)*. arXiv:2611.09501. <https://arxiv.org/abs/2611.09501>
#'
#' Reference implementation: `faiss.contrib.e_means`,
#' <https://github.com/facebookresearch/faiss/tree/main/contrib/e_means>
#'
#' @seealso [shoal_kmeans()] for Lloyd's algorithm with a k-means++ start,
#'   and [predict.shoal_kmeans()] for assigning new observations.
#'
#' @examples
#' set.seed(1)
#' centres <- matrix(rnorm(40 * 8, sd = 4), ncol = 8)
#' x <- centres[sample.int(40, 20000, replace = TRUE), ] +
#'   matrix(rnorm(20000 * 8), ncol = 8)
#'
#' fit <- shoal_emeans(x, k = 40L)
#' fit
#' fit$recipe$batch_size
#' predict(fit, x[1:5, ])
#'
#' # Spherical k-means, clustering directions.
#' shoal_emeans(x, k = 40L, metric = "cosine")
#'
#' @export
shoal_emeans <- function(x, k, epochs = 5,
                         metric = c("euclidean", "cosine"),
                         batch_law = c("auto", "throughput", "quality"),
                         batch_size = NULL, alpha0 = NULL,
                         revive_threshold = NULL, seed = 1L) {
  rlang::check_required(k)
  x <- check_numeric_matrix(x)
  check_positive_integer(k)
  check_positive_number(epochs)
  metric <- rlang::arg_match(metric)
  batch_law <- rlang::arg_match(batch_law)
  if (!is.null(batch_size)) {
    check_positive_integer(batch_size)
  }
  if (!is.null(alpha0)) {
    check_positive_number(alpha0)
    if (alpha0 > 1) {
      cli::cli_abort("{.arg alpha0} must be at most 1.")
    }
  }
  if (!is.null(revive_threshold)) {
    check_non_negative_number(revive_threshold)
  }
  check_count(seed)

  n <- nrow(x)
  if (k > n) {
    cli::cli_abort("{.arg k} ({k}) cannot exceed the number of rows ({n}).")
  }
  if (epochs * n < 1) {
    cli::cli_abort(c(
      "{.arg epochs} must cover at least one row.",
      "i" = "{.arg epochs} ({epochs}) times the number of rows ({n}) is below 1."
    ))
  }

  # Settings left to the recipe cross as sentinels: 0 for the batch size,
  # NaN (not NA, which the conversion refuses) for the two numbers.
  result <- rust_emeans(
    x, as.integer(k), as.double(epochs), batch_law,
    if (is.null(batch_size)) 0L else as.integer(min(batch_size, n)),
    if (is.null(alpha0)) NaN else as.double(alpha0),
    if (is.null(revive_threshold)) NaN else as.double(revive_threshold),
    identical(metric, "cosine"), as.double(seed)
  )
  check_no_zero_rows(result$zero_rows, "x")

  colnames(result$centroids) <- colnames(x)

  # Overrides are recorded only when given, so the printed parameters show
  # the call; the resolved values are in `recipe` either way.
  params <- list(
    k = as.integer(k),
    epochs = epochs,
    batch_law = batch_law,
    batch_size = if (!is.null(batch_size)) as.integer(batch_size),
    alpha0 = alpha0,
    revive_threshold = revive_threshold,
    # Kept as double: a whole-number seed can exceed the integer range.
    seed = as.numeric(seed)
  )
  params <- params[!vapply(params, is.null, logical(1L))]

  new_clustering(
    cluster = result$cluster,
    data = x,
    algorithm = "e-means",
    subclass = c("shoal_emeans", "shoal_kmeans"),
    params = params,
    metric = metric,
    centroids = result$centroids,
    inertia = result$inertia,
    sizes = as.integer(result$sizes),
    epoch_objective = result$epoch_objective,
    n_splits = as.integer(result$n_splits),
    recipe = list(
      batch_law = result$batch_law,
      batch_size = as.integer(result$batch_size),
      alpha0 = result$alpha0,
      epoch_rows = result$epoch_rows,
      total_steps = result$total_steps,
      revive_threshold = result$revive
    )
  )
}

#' Abort when cosine mode met rows with no direction
#' @noRd
check_no_zero_rows <- function(zero_rows, arg, call = rlang::caller_env()) {
  if (zero_rows > 0) {
    cli::cli_abort(
      c(
        "{.arg {arg}} has {zero_rows} all-zero row{?s}.",
        "i" = "Cosine mode clusters directions, and an all-zero row has none."
      ),
      call = call
    )
  }
  invisible(NULL)
}

#' @rdname print.shoal
#' @export
print.shoal_emeans <- function(x, ...) {
  # The shared heading, then the k-means summary with the sizes condensed:
  # e-means is for large k, where listing every size would fill the console.
  print.shoal_clustering(x)

  sizes <- x$sizes
  shown <- if (length(sizes) <= 10L) {
    paste(sizes, collapse = ", ")
  } else {
    sprintf(
      "min %d, median %s, max %d",
      min(sizes), format(stats::median(sizes)), max(sizes)
    )
  }
  empty <- sum(sizes == 0L)
  cli::cli_text("Within-cluster sum of squares: {signif(x$inertia, 5L)}")
  cli::cli_text(
    "Cluster sizes: {shown}{if (empty > 0L) sprintf(' (%d empty)', empty) else ''}"
  )

  r <- x$recipe
  # Numeric, not formatted: cli pluralises on the value.
  passes <- signif(sum(r$epoch_rows) / nrow(x$data), 4L)
  cli::cli_text(
    "Training: {passes} epoch{?s}, {r$total_steps} updates of {r$batch_size} rows ",
    "({r$batch_law} batch law), initial step {signif(r$alpha0, 3L)}, ",
    "{x$n_splits} cluster{?s} revived"
  )

  invisible(x)
}

#' @rdname predict.shoal_kmeans
#' @export
predict.shoal_emeans <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(object$cluster)
  }

  newdata <- check_numeric_matrix(newdata, na_action = "error")
  check_newdata_columns(newdata, object$centroids)

  res <- rust_emeans_assign(
    newdata, object$centroids, identical(object$metric, "cosine")
  )
  check_no_zero_rows(res$zero_rows, "newdata")
  res$cluster
}
