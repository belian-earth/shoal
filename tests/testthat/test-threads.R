test_that("shoal_threads reports and sets the pool size", {
  old <- shoal_threads()
  on.exit(shoal_threads(old))
  expect_gte(old, 1L)

  expect_invisible(shoal_threads(2))
  expect_identical(shoal_threads(), 2L)
  shoal_threads(1)
  expect_identical(shoal_threads(), 1L)

  expect_error(shoal_threads(0), "positive integer")
  expect_error(shoal_threads("many"), "single integer")
})

test_that("results do not depend on the thread count", {
  old <- shoal_threads()
  on.exit(shoal_threads(old))
  x <- as.matrix(iris[, 1:4])
  set.seed(1)
  emb <- matrix(runif(4 * 32, -1, 1) * 0.6, nrow = 4)[rep(1:4, each = 60L), ] +
    matrix(rnorm(240 * 32, sd = 0.1), ncol = 32)
  # Enough rows for several of the k-means|| sampling blocks of 4096.
  wide <- matrix(rnorm(12000 * 3), ncol = 3)

  run <- function() list(
    dist = shoal_dist(x),
    dbscan = shoal_dbscan(x, eps = 0.5, min_samples = 5L)$cluster,
    hdbscan = shoal_hdbscan(x)$cluster,
    evoc = shoal_evoc(emb, min_cluster_size = 15L)$layers,
    kmeans = shoal_kmeans(x, k = 3L)[c("cluster", "centroids", "inertia")],
    kmeans_par = shoal_kmeans(wide, k = 20L, init = "kmeans_parallel", n_runs = 2L,
                              max_iter = 5L)[c("cluster", "centroids", "inertia")],
    # A batch of several assignment tiles, so training runs in parallel too.
    emeans = shoal_emeans(emb, k = 8L, batch_size = 256L)[c("cluster", "centroids", "inertia", "epoch_objective")],
    emeans_cos = shoal_emeans(emb, k = 8L, metric = "cosine")[c("cluster", "centroids", "inertia")],
    gmm = shoal_gmm(x, k = 3L)[c("cluster", "means", "loglik")],
    sil = shoal_silhouette(shoal_dist(x), shoal_kmeans(x, k = 3L))$width
  )
  shoal_threads(1)
  one <- run()
  shoal_threads(4)
  four <- run()
  expect_identical(one, four)
})

test_that("the load-time default follows option, then variable, then cores", {
  old <- shoal_threads()
  on.exit(shoal_threads(old))

  withr::with_options(list(shoal.threads = 5), expect_identical(default_threads(), 5L))
  withr::with_envvar(c(RAYON_NUM_THREADS = "3"), expect_identical(default_threads(), 3L))
  withr::with_envvar(c(RAYON_NUM_THREADS = "", "_R_CHECK_LIMIT_CORES_" = "TRUE"),
                     expect_lte(default_threads(), 2L))
  withr::with_envvar(c(RAYON_NUM_THREADS = "", "_R_CHECK_LIMIT_CORES_" = ""), {
    withr::with_options(list(shoal.threads = NULL), {
      expect_identical(default_threads(), max(parallel::detectCores(logical = TRUE), 1L))
    })
  })

  # .onLoad applies the default to the pool. pkgload does not keep the hook
  # in the namespace, so this part runs only against an installed package.
  onload <- get0(".onLoad", envir = asNamespace("shoal"), inherits = FALSE)
  skip_if(is.null(onload), ".onLoad not present under load_all()")
  withr::with_options(list(shoal.threads = 2), onload(NULL, "shoal"))
  expect_identical(shoal_threads(), 2L)
})
