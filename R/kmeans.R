#' K-Means Clustering
#'
#' Lloyd's algorithm via the Rust
#' \href{https://github.com/rust-ml/linfa}{linfa} toolkit, with k-means++ and
#' k-means|| initialisation.
#'
#' Unlike the density-based algorithms, k-means partitions every observation:
#' there is no noise class, so `cluster` never contains `NA`. It is also
#' predictive: [predict()] assigns new observations to the fitted centroids.
#'
#' # Reproducibility
#'
#' k-means is stochastic in its initialisation, so `seed` is a parameter rather
#' than being taken from R's RNG. The same `seed` and parameters always give the
#' same partition; `set.seed()` has no effect on it.
#'
#' # Choosing k
#'
#' `inertia` (the within-cluster sum of squares) is returned so it can be
#' compared across values of `k`, the usual scree or elbow approach. Note that
#' it decreases monotonically with `k` by construction, so it identifies a
#' diminishing-returns point rather than an optimum.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped).
#' @param k Number of clusters. Required, since there is no sensible default for
#'   the central modelling decision. Must be at least 1 and no more than `nrow(x)`.
#' @param init Initialisation method. `"kmeans++"` (default) is the usual
#'   choice; `"kmeans_parallel"` scales better past roughly 100 clusters;
#'   `"random"` is the naive baseline.
#' @param n_runs Number of restarts, keeping the fit with the lowest inertia.
#'   Default `10L`.
#' @param max_iter Maximum iterations per run. Default `300L`.
#' @param tolerance Convergence threshold on centroid movement. Default `1e-4`.
#' @param seed Non-negative whole-number seed for initialisation. Stored and
#'   passed as a double, so values beyond the integer range are safe. Default `1L`.
#'
#' @returns An object of class `c("shoal_kmeans", "shoal_clustering")`: a list
#'   with components `cluster` (integer vector of cluster IDs), `n_clusters`,
#'   `n_noise` (always `0`), `data`, `algorithm`, `params`, `centroids`
#'   (a `k x ncol(x)` matrix), `inertia` and `sizes`.
#'
#' @seealso [predict.shoal_kmeans()] for assigning new observations, and
#'   [shoal_emeans()] for many clusters on large data.
#'
#' @examples
#' fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L)
#' fit
#' fit$centroids
#'
#' @export
shoal_kmeans <- function(x, k,
                         init = c("kmeans++", "kmeans_parallel", "random"),
                         n_runs = 10L, max_iter = 300L, tolerance = 1e-4,
                         seed = 1L) {
  rlang::check_required(k)
  x <- check_numeric_matrix(x)
  check_positive_integer(k)
  init <- rlang::arg_match(init)
  check_positive_integer(n_runs)
  check_positive_integer(max_iter)
  check_positive_number(tolerance)
  check_count(seed)

  if (k > nrow(x)) {
    cli::cli_abort("{.arg k} ({k}) cannot exceed the number of rows ({nrow(x)}).")
  }

  result <- rust_kmeans(
    x, as.integer(k), init, as.integer(n_runs), as.integer(max_iter),
    as.double(tolerance), as.double(seed)
  )

  colnames(result$centroids) <- colnames(x)

  new_clustering(
    cluster = result$cluster,
    data = x,
    algorithm = "K-Means",
    subclass = "shoal_kmeans",
    params = list(
      k = as.integer(k),
      init = init,
      n_runs = as.integer(n_runs),
      # Kept as double: a whole-number seed can exceed the integer range.
      seed = as.numeric(seed)
    ),
    centroids = result$centroids,
    inertia = result$inertia,
    sizes = as.integer(result$sizes)
  )
}

#' Assign New Observations to Fitted Clusters
#'
#' Assigns each row of `newdata` to the nearest centroid by squared Euclidean
#' distance, or for a cosine [shoal_emeans()] model to the centroid of largest
#' cosine similarity. For a [shoal_emeans()] model, predicting the training
#' data on the machine that fitted it reproduces its `cluster` exactly.
#'
#' Only algorithms with a notion of a cluster centre can do this. The
#' density-based methods deliberately have no `predict()` method, so calling
#' [predict()] on their results raises R's standard "no applicable method"
#' error rather than a bespoke one.
#'
#' @param object A fitted [shoal_kmeans()] or [shoal_emeans()] model.
#' @param newdata A numeric matrix or data frame with the same columns as the
#'   data the model was fitted to. Omit it to return the training assignments.
#' @param ... Ignored.
#'
#' @returns An integer vector of cluster IDs, one per row of `newdata`.
#'
#' @examples
#' fit <- shoal_kmeans(as.matrix(iris[1:100, 1:4]), k = 2L)
#' predict(fit, as.matrix(iris[101:150, 1:4]))
#'
#' @export
predict.shoal_kmeans <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(object$cluster)
  }

  newdata <- check_numeric_matrix(newdata, na_action = "error")
  check_newdata_columns(newdata, object$centroids)

  rust_nearest_centroid(newdata, object$centroids)
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
  check_finite_distances(res$finite, "newdata")
  res$cluster
}
