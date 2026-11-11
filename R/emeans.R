#' e-means Clustering
#'
#' k-means fitted by e-means (Bessa et al. 2026), via an in-tree Rust port
#' of the reference implementation in
#' \href{https://github.com/facebookresearch/faiss/tree/main/contrib/e_means}{faiss}
#' (Meta FAIR). Lloyd's algorithm, behind [shoal_kmeans()], recomputes every
#' centroid from all the data at each iteration. e-means instead visits the
#' data in random mini-batches and moves each centroid towards its batch
#' mean by an exponential moving average. The batch size and step size come
#' from scaling laws the authors fitted on the number of rows, clusters and
#' passes. It minimises the same within-cluster sum of squares, and with
#' many rows and hundreds of clusters or more it reaches the loss of a
#' converged k-means in a handful of passes over the data.
#'
#' # When to use it
#'
#' e-means is for scale: many rows and many clusters. Typical uses are a
#' codebook for vector quantisation, a fixed set of centroids that stands in
#' for every vector nearest to it, and a fine partition of an embedding
#' collection. The scaling laws were fitted on \eqn{10^5} to \eqn{10^8} rows
#' and 256 to 65,536 clusters. Outside that range e-means still runs, but
#' its settings are not tuned there. On small data with few clusters, prefer
#' [shoal_kmeans()]. Its restarts, ten by default, guard against the poor
#' local minima that a single random start can settle in, and e-means makes
#' one random start.
#'
#' The cost is fixed by `epochs`: one pass over the data per epoch, plus one
#' for the final assignment. There are no restarts and no convergence test.
#'
#' Results do not depend on where the data sits. Euclidean mode works
#' relative to the column means, so projected coordinates far from the
#' origin cluster as well as local ones.
#'
#' # Cosine mode
#'
#' `metric = "cosine"` is spherical k-means, for embedding vectors compared
#' by cosine similarity. Rows are scaled to unit length internally, each is
#' assigned to the centroid of largest cosine similarity, and the centroids
#' are unit vectors. Rescaling rows therefore has no effect. A row of all
#' zeros has no direction and is an error. `inertia` is then the sum of
#' squared distances between the unit rows and their unit centroids,
#' \eqn{\sum 2(1 - \cos\theta)}.
#'
#' # The training plan
#'
#' Unless given, the batch size and the initial step `alpha0` come from the
#' fitted laws. The step then falls along half a cosine curve to a tenth of
#' `alpha0` by the end of the run. `batch_law = "quality"` uses batches
#' about eight times smaller than `"throughput"`, trading speed for a lower
#' final loss; `"auto"` chooses `"quality"` for data with at most 16
#' columns and `"throughput"` above, as upstream does.
#'
#' Each centroid carries a running mass, a moving average of the number of
#' rows assigned to it. Every five updates, clusters whose mass has fallen
#' below `revive_threshold` times the mean mass are revived: a donor drawn
#' in proportion to its mass is split in two, and one half takes the place
#' of the starved cluster. The plan used is returned in `recipe`.
#'
#' # Assignments
#'
#' The centroids are the means of the rows assigned to them during the final
#' training epoch, scaled to unit length in cosine mode. One exact pass then
#' assigns each row to its nearest final centroid, so `cluster`, `inertia`
#' and `sizes` describe the partition the centroids induce, and [predict()]
#' on the training data reproduces `cluster`. The reference implementation
#' saves that pass and returns the labels from the last training epoch,
#' during which the centroids were still moving. A centroid can end the run
#' with no points. `sizes` then has a zero for it, and `n_clusters` counts
#' only the clusters with points.
#'
#' # Reproducibility
#'
#' As in [shoal_kmeans()], `seed` is a parameter rather than being taken
#' from R's RNG. The same seed, parameters and data give identical results
#' whatever the thread count. Between machines, results can differ by
#' rounding, because the matrix-product kernel is chosen for the CPU, and a
#' row at a near-tie between two centroids can then change cluster. A seed
#' does not reproduce a run of the reference implementation, whose random
#' streams differ; the port is checked against it with the start, the batch
#' order and the random draws fixed instead.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a
#'   matrix using their numeric columns (non-numeric columns are dropped).
#' @param k Number of clusters. Must be at least 1 and no more than
#'   `nrow(x)`.
#' @param epochs Number of passes over the data, the training budget, at
#'   most 1000. A fractional value visits a random fraction of the rows in
#'   one of the passes. Default `5`, as upstream.
#' @param metric `"euclidean"` (default) for k-means, or `"cosine"` for
#'   spherical k-means on directions; see the Cosine mode section.
#' @param batch_law Which fitted law sets the batch size when `batch_size`
#'   is not given: `"auto"` (default), `"throughput"` or `"quality"`.
#' @param batch_size Rows per update, capped at `nrow(x)`. `NULL` (default)
#'   uses the batch law.
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
#'   `epoch_objective`, `n_splits` and `recipe`.
#'
#'   `epoch_objective` gives, per epoch, the sum of squared distances from
#'   the rows visited to the centroids they were assigned against. Those
#'   centroids move during the epoch, so it is a record of training, not the
#'   final error. `n_splits` counts revivals, and a cluster can be revived
#'   more than once. `recipe` is the plan used: `batch_law`, which is `NA`
#'   when `batch_size` was given, `batch_size`, `alpha0`, `epoch_rows` (rows
#'   visited per epoch), `total_steps` and `revive_threshold`.
#'
#' @references
#' Bessa, S., Mazaré, P.-E., Szilvasy, G., Touvron, H., Picard, D., Douze, M.
#' and Jégou, H. (2026). *e-means: outperforming k-means (in one-tenth the
#' epochs)*. arXiv:2611.09501. <https://arxiv.org/abs/2611.09501>
#'
#' Reference implementation: `faiss.contrib.e_means`,
#' <https://github.com/facebookresearch/faiss/tree/main/contrib/e_means>
#'
#' @seealso [shoal_kmeans()] for Lloyd's algorithm with k-means++ restarts,
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
  # The plan holds one entry per epoch; far beyond any useful budget, it
  # would exhaust memory before training began.
  if (epochs > 1000) {
    cli::cli_abort("{.arg epochs} must be at most 1000, not {epochs}.")
  }
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
  # A batch larger than the data is the data; capping first keeps the value
  # within the integer range.
  if (!is.null(batch_size)) {
    batch_size <- as.integer(min(batch_size, n))
  }

  # Settings left to the recipe cross as sentinels: 0 for the batch size,
  # NaN (not NA, which the conversion refuses) for the two numbers.
  result <- rust_emeans(
    x, as.integer(k), as.double(epochs), batch_law,
    batch_size %||% 0L,
    if (is.null(alpha0)) NaN else as.double(alpha0),
    if (is.null(revive_threshold)) NaN else as.double(revive_threshold),
    identical(metric, "cosine"), as.double(seed)
  )
  check_no_zero_rows(result$zero_rows, "x")
  check_finite_distances(result$finite, "x")

  colnames(result$centroids) <- colnames(x)

  # Overrides are recorded only when given, so the printed parameters show
  # the call; the resolved values are in `recipe` either way.
  params <- list(
    k = as.integer(k),
    epochs = epochs,
    batch_law = batch_law,
    batch_size = batch_size,
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
      batch_law = if (nzchar(result$batch_law)) result$batch_law else NA_character_,
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

#' Abort when squared distances overflowed double precision
#' @noRd
check_finite_distances <- function(finite, arg, call = rlang::caller_env()) {
  if (!isTRUE(finite)) {
    cli::cli_abort(
      c(
        "Squared distances in {.arg {arg}} overflowed double precision.",
        "i" = "Rescale {.arg {arg}}: a common scale factor does not change the clusters."
      ),
      call = call
    )
  }
  invisible(NULL)
}
