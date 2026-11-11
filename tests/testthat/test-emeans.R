blobs <- function(n = 5000L, p = 8L, groups = 40L, seed = 2L) {
  withr::with_seed(seed, {
    centres <- matrix(stats::rnorm(groups * p, sd = 4), ncol = p)
    centres[sample.int(groups, n, replace = TRUE), ] + matrix(stats::rnorm(n * p), ncol = p)
  })
}

test_that("shoal_emeans partitions every observation", {
  x <- blobs()
  fit <- shoal_emeans(x, k = 40L)

  expect_s3_class(fit, "shoal_emeans")
  expect_s3_class(fit, "shoal_kmeans")
  expect_s3_class(fit, "shoal_clustering")
  expect_length(fit$cluster, nrow(x))
  expect_false(anyNA(fit$cluster))
  expect_equal(fit$n_noise, 0L)
  expect_equal(fit$metric, "euclidean")
  expect_equal(fit$n_clusters, length(unique(fit$cluster)))
})

test_that("cluster, sizes and inertia describe the partition the centroids induce", {
  x <- blobs()
  fit <- shoal_emeans(x, k = 40L)

  expect_equal(dim(fit$centroids), c(40L, ncol(x)))
  expect_equal(sum(fit$sizes), nrow(x))
  expect_equal(fit$sizes, tabulate(fit$cluster, nbins = 40L))

  wss <- sum((x - fit$centroids[fit$cluster, ])^2)
  expect_equal(fit$inertia, wss, tolerance = 1e-9)

  # Every row's label is its nearest final centroid, so predicting the
  # training data reproduces the assignment exactly.
  expect_identical(predict(fit, x), fit$cluster)
  expect_identical(predict(fit), fit$cluster)
  d2 <- outer(rowSums(x^2), rep(1, 40L)) - 2 * x %*% t(fit$centroids) +
    outer(rep(1, nrow(x)), rowSums(fit$centroids^2))
  expect_equal(fit$cluster, max.col(-d2, ties.method = "first"))
})

test_that("five epochs match a converged k-means on unstructured data", {
  # No cluster structure, so no local minimum for a single start to fall
  # into: the comparison is of the optimisers alone.
  x <- withr::with_seed(3, matrix(stats::rnorm(5000 * 8), ncol = 8))
  em <- shoal_emeans(x, k = 40L)
  km <- shoal_kmeans(x, k = 40L, n_runs = 3L)
  expect_lt(em$inertia / km$inertia, 1.02)
})

test_that("the seed makes runs reproducible", {
  x <- blobs(n = 2000L)
  a <- shoal_emeans(x, k = 20L, seed = 42L)
  b <- shoal_emeans(x, k = 20L, seed = 42L)
  expect_identical(a$cluster, b$cluster)
  expect_identical(a$centroids, b$centroids)
  expect_false(identical(a$centroids, shoal_emeans(x, k = 20L, seed = 43L)$centroids))

  # R's own RNG state is irrelevant to it.
  c1 <- withr::with_seed(1, shoal_emeans(x, k = 20L, seed = 7L)$cluster)
  c2 <- withr::with_seed(999, shoal_emeans(x, k = 20L, seed = 7L)$cluster)
  expect_identical(c1, c2)
})

test_that("where the data sits does not change the fit", {
  # Projected coordinates sit far from the origin.
  x <- blobs(n = 4000L, p = 3L, groups = 60L)
  far <- sweep(x, 2, c(4.5e5, 5.6e6, 300), "+")
  near <- shoal_emeans(x, k = 64L)
  shifted <- shoal_emeans(far, k = 64L)
  expect_identical(shifted$cluster, near$cluster)
  expect_equal(shifted$inertia, near$inertia, tolerance = 1e-6)
  expect_identical(predict(shifted, far), shifted$cluster)
})

test_that("the recipe follows the epoch budget and the overrides", {
  x <- blobs(n = 3000L)
  fit <- shoal_emeans(x, k = 30L, epochs = 2.5)
  expect_equal(fit$recipe$epoch_rows, c(3000, 1500, 3000))
  expect_length(fit$epoch_objective, 3L)
  # Eight columns: the automatic law is the quality law.
  expect_equal(fit$recipe$batch_law, "quality")
  expect_equal(fit$recipe$total_steps, sum(ceiling(fit$recipe$epoch_rows / fit$recipe$batch_size)))

  part <- shoal_emeans(x, k = 30L, epochs = 0.5)
  expect_equal(part$recipe$epoch_rows, 1500)

  thr <- shoal_emeans(x, k = 30L, batch_law = "throughput")
  qual <- shoal_emeans(x, k = 30L, batch_law = "quality")
  expect_gt(thr$recipe$batch_size, qual$recipe$batch_size)

  set <- shoal_emeans(x, k = 30L, batch_size = 100L, alpha0 = 0.3, revive_threshold = 0)
  expect_equal(set$recipe$batch_size, 100L)
  expect_equal(set$recipe$alpha0, 0.3)
  expect_equal(set$recipe$revive_threshold, 0)
  # No law chose a given batch size.
  expect_identical(set$recipe$batch_law, NA_character_)
  expect_equal(set$n_splits, 0L)
  expect_equal(set$params$batch_size, 100L)
  expect_null(fit$params$batch_size)

  # A batch larger than the data is the data, beyond the integer range too.
  expect_no_warning(big <- shoal_emeans(x, k = 3L, batch_size = 1e10))
  expect_equal(big$recipe$batch_size, 3000L)
  expect_equal(big$params$batch_size, 3000L)
  # A full step replaces each centroid by its batch mean.
  expect_no_error(shoal_emeans(x, k = 3L, alpha0 = 1))
  # Revival is off by default at two columns, as upstream.
  expect_equal(shoal_emeans(x[, 1:2], k = 5L)$recipe$revive_threshold, 0)
})

test_that("revival restarts starved clusters and leaves empty ones visible", {
  # Five distinct points for eight clusters: the surplus starve, are
  # revived on top of the points, and some end with nothing.
  x <- blobs(n = 5L)[rep(1:5, 40L), ]
  fit <- shoal_emeans(x, k = 8L)
  expect_gt(fit$n_splits, 0L)
  expect_equal(fit$n_clusters, 5L)
  expect_equal(sum(fit$sizes == 0L), 3L)
  expect_equal(sum(fit$sizes), 200L)
})

test_that("cosine mode clusters directions", {
  x <- blobs(n = 3000L)
  fit <- shoal_emeans(x, k = 25L, metric = "cosine")
  expect_equal(fit$metric, "cosine")
  expect_equal(unname(sqrt(rowSums(fit$centroids^2))), rep(1, 25L), tolerance = 1e-12)

  # Inertia is measured between unit vectors.
  u <- x / sqrt(rowSums(x^2))
  expect_equal(fit$inertia, sum((u - fit$centroids[fit$cluster, ])^2), tolerance = 1e-9)

  # Row scale is irrelevant: powers of two rescale exactly.
  scale <- rep_len(c(0.25, 1, 8), nrow(x))
  rescaled <- shoal_emeans(x * scale, k = 25L, metric = "cosine")
  expect_identical(rescaled$cluster, fit$cluster)
  expect_identical(rescaled$centroids, fit$centroids)
  expect_identical(predict(fit, x * scale), fit$cluster)

  # The largest cosine similarity picks the cluster.
  expect_equal(fit$cluster, max.col(u %*% t(fit$centroids), ties.method = "first"))
})

test_that("cosine mode refuses rows with no direction", {
  x <- blobs(n = 500L)
  fit <- shoal_emeans(x, k = 5L, metric = "cosine")
  x[c(3, 9), ] <- 0
  expect_error(shoal_emeans(x, k = 5L, metric = "cosine"), "2 all-zero rows")
  expect_error(predict(fit, x), "2 all-zero rows")
  # Euclidean mode has no such restriction.
  expect_no_error(shoal_emeans(x, k = 5L))
})

test_that("overflowing distances are an error, not a silent partition", {
  x <- blobs(n = 200L) * 1e160
  expect_error(shoal_emeans(x, k = 4L), "overflowed")
  fit <- shoal_emeans(blobs(n = 200L), k = 4L)
  expect_error(predict(fit, x), "overflowed")
})

test_that("k = 1 and k = nrow(x) are both handled", {
  x <- blobs(n = 60L)

  one <- shoal_emeans(x, k = 1L)
  expect_equal(unique(one$cluster), 1L)
  expect_equal(unname(one$centroids[1, ]), unname(colMeans(x)), tolerance = 1e-12)

  all_rows <- shoal_emeans(x, k = 60L)
  expect_equal(nrow(all_rows$centroids), 60L)
  expect_equal(sum(all_rows$sizes), 60L)
})

test_that("data frames and incomplete rows are accepted", {
  df <- as.data.frame(blobs(n = 400L))
  df$label <- "a"
  fit <- shoal_emeans(df, k = 4L)
  expect_equal(ncol(fit$data), 8L)
  expect_equal(colnames(fit$centroids), paste0("V", 1:8))
  expect_identical(predict(fit, df[1:9, ]), fit$cluster[1:9])

  m <- blobs(n = 400L)
  m[5, 2] <- NA
  expect_warning(fit <- shoal_emeans(m, k = 4L), "Removed 1 row")
  expect_length(fit$cluster, 399L)
})

test_that("shoal_emeans validates its inputs", {
  x <- blobs(n = 20L)

  expect_error(shoal_emeans(x), "absent")
  expect_error(shoal_emeans(x, k = 0L), "positive integer")
  expect_error(shoal_emeans(x, k = 21L), "cannot exceed")
  expect_error(shoal_emeans(x, k = 2L, epochs = 0), "positive")
  expect_error(shoal_emeans(x, k = 2L, epochs = 0.01), "at least one row")
  expect_error(shoal_emeans(x, k = 2L, epochs = 1e12), "at most 1000")
  expect_error(shoal_emeans(x, k = 2L, metric = "manhattan"))
  expect_error(shoal_emeans(x, k = 2L, batch_law = "fast"))
  expect_error(shoal_emeans(x, k = 2L, batch_size = 0L), "positive integer")
  expect_error(shoal_emeans(x, k = 2L, alpha0 = 0), "positive")
  expect_error(shoal_emeans(x, k = 2L, alpha0 = 1.5), "at most 1")
  expect_error(shoal_emeans(x, k = 2L, revive_threshold = -1), "non-negative")
  expect_error(shoal_emeans(x, k = 2L, revive_threshold = NA_real_), "non-negative")
  expect_error(shoal_emeans(x, k = 2L, seed = -1L), "non-negative")
  expect_error(shoal_emeans(1:10, k = 2L), "numeric matrix or data frame")
})

test_that("predict checks newdata", {
  x <- blobs(n = 300L)
  colnames(x) <- paste0("b", 1:8)
  fit <- shoal_emeans(x, k = 4L)
  expect_error(predict(fit, x[, 1:3]), "fitted on 8")
  expect_length(predict(fit, x[1:7, ]), 7L)
  renamed <- x
  colnames(renamed) <- paste0("c", 1:8)
  expect_warning(predict(fit, renamed), "differ")
  bad <- x[1:3, ]
  bad[2, 1] <- NA
  expect_error(predict(fit, bad), "non-finite")
})
