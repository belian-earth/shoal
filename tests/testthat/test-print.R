test_that("print methods report each algorithm's extra components", {
  x <- as.matrix(iris[, 1:4])
  # The methods print through cli, which emits messages rather than stdout.
  # Whitespace is squashed so a pattern does not depend on where cli wraps.
  printed <- function(fit) gsub("\\s+", " ", paste(cli::cli_fmt(print(fit)), collapse = "\n"))

  km <- shoal_kmeans(x, k = 3L)
  expect_match(printed(km), "K-Means Clustering")
  expect_match(printed(km), "Within-cluster sum of squares")
  expect_match(printed(km), "Cluster sizes")

  gm <- shoal_gmm(x, k = 2L)
  expect_match(printed(gm), "Gaussian Mixture Clustering")
  expect_match(printed(gm), "Log-likelihood")
  expect_match(printed(gm), "Mixing proportions")

  set.seed(1)
  emb <- matrix(runif(3 * 32, -1, 1) * 0.6, nrow = 3)[rep(1:3, each = 60L), ] +
    matrix(rnorm(180 * 32, sd = 0.1), ncol = 32)
  ev <- shoal_evoc(emb, min_cluster_size = 15L)
  expect_match(printed(ev), "EVoC Clustering")
  expect_match(printed(ev), "Layers \\(finest first")
  expect_match(printed(ev), "persistence")

  em <- shoal_emeans(emb, k = 4L)
  expect_match(printed(em), "e-means Clustering")
  expect_match(printed(em), "Within-cluster sum of squares")
  expect_match(printed(em), "Cluster sizes: [0-9]+, ")
  expect_match(printed(em), "Training: 5 epochs, [0-9]+ updates of [0-9]+ rows \\(throughput batch law\\)")
  expect_match(printed(em), "[0-9]+ revivals?$")
  given <- shoal_emeans(emb, k = 4L, batch_size = 1L, epochs = 1)
  expect_match(printed(given), "updates of 1 row \\(batch size given\\)")
  dup <- shoal_emeans(emb[rep(1:5, 20L), ], k = 8L)
  expect_match(printed(dup), "Cluster sizes: .*\\([0-9]+ empty\\)")
  # Many clusters: the sizes are summarised, not listed.
  em_many <- shoal_emeans(emb, k = 30L, metric = "cosine")
  expect_match(printed(em_many), "Metric: .*cosine")
  expect_match(printed(em_many), "Cluster sizes: min [0-9]+, median")

  # print() returns its argument invisibly for every method.
  for (fit in list(km, gm, ev, em)) {
    expect_invisible(print(fit))
    expect_identical(withVisible(print(fit))$value, fit)
  }
})
