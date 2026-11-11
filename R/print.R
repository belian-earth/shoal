#' Print a clustering result
#'
#' A single method serves every algorithm: the heading, parameters and cluster
#' counts all come from components the shared constructor guarantees. Algorithms
#' with extra output specialise and call [NextMethod()].
#'
#' @param x A clustering result.
#' @param ... Ignored.
#'
#' @returns `x`, invisibly.
#'
#' @name print.shoal
NULL

#' @rdname print.shoal
#' @export
print.shoal_clustering <- function(x, ...) {
  params <- format_params(x$params)

  cli::cli_h3("{x$algorithm} Clustering")
  if (!is.null(x$metric)) {
    cli::cli_text("Metric: {.val {x$metric}}")
  }
  cli::cli_text("Parameters: {params}")
  cli::cli_text("Clusters: {x$n_clusters}, Noise points: {x$n_noise}")
  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_hdbscan <- function(x, ...) {
  NextMethod()

  med <- round(stats::median(x$outlier_scores), 3L)
  mx <- round(max(x$outlier_scores), 3L)
  cli::cli_text("GLOSH outlier scores: median {med}, max {mx}")

  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_evoc <- function(x, ...) {
  NextMethod()

  cli::cli_text("Layers (finest first, {cli::symbol$tick} = selected):")
  lines <- vapply(seq_along(x$layers), function(i) {
    l <- x$layers[[i]]
    k <- length(unique(l[!is.na(l)]))
    sprintf(
      "%d: %d cluster%s, %d noise, persistence %s",
      i, k, if (k == 1L) "" else "s", sum(is.na(l)),
      format(signif(x$persistence[[i]], 4L))
    )
  }, character(1L))
  # Bullet names control the marker: a tick for the selected layer and an
  # aligned blank for the rest, which cli_text's whitespace collapsing would
  # otherwise strip.
  names(lines) <- ifelse(seq_along(lines) == x$layer, "v", " ")
  cli::cli_bullets(lines)

  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_kmeans <- function(x, ...) {
  NextMethod()

  inertia <- signif(x$inertia, 5L)
  sizes <- paste(x$sizes, collapse = ", ")
  cli::cli_text("Within-cluster sum of squares: {inertia}")
  cli::cli_text("Cluster sizes: {sizes}")

  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_gmm <- function(x, ...) {
  NextMethod()

  loglik <- signif(x$loglik, 6L)
  bic <- signif(stats::BIC(x), 6L)
  weights <- paste(signif(x$weights, 3L), collapse = ", ")
  cli::cli_text("Log-likelihood: {loglik}, BIC: {bic}")
  cli::cli_text("Mixing proportions: {weights}")

  invisible(x)
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
  law <- if (is.na(r$batch_law)) "batch size given" else paste(r$batch_law, "batch law")
  cli::cli_text(
    "Training: {passes} epoch{?s}, {r$total_steps} update{?s} of ",
    "{r$batch_size} row{?s} ({law}), initial step {signif(r$alpha0, 3L)}, ",
    "{x$n_splits} revival{?s}"
  )

  invisible(x)
}
