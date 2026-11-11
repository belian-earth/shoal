# Benchmark: shoal_emeans() against k-means in R, at the scale e-means is for.
# Run from project root:
#   NOT_CRAN=true R CMD INSTALL .        # release build
#   Rscript bench/bench_emeans.R         # data, the R runs and their centroids
#   uv run bench/bench_emeans.py         # the Python runs, then scores everything
#
# The scaling benchmark in bench_r.R uses five clusters and at most 50,000
# rows, where e-means has nothing to offer. This one uses 500,000 rows of
# embedding-like data in 64 dimensions and 256 or 1024 clusters. Speed alone
# would flatter a method that stops early, so every method's centroids are
# saved and scored on the same held-out rows by bench_emeans.py.
#
# Settings: one start for every method, from k random rows, as e-means and
# faiss start. Lloyd-type methods run at most 25 iterations, the default of
# faiss's k-means; mini-batch methods run 5 passes with the batch size e-means
# chose. Runs over 30 s are timed once, faster ones take the median of three.
# The k-means++ start of shoal_kmeans() is timed separately: its cost grows
# with the square of k, and at k = 1024 it alone would take minutes.

library(shoal)

d <- 64L
n_train <- 500000L
n_test <- 50000L
ks <- c(256L, 1024L)
lloyd_iters <- 25L
dir.create("bench/data/emeans", recursive = TRUE, showWarnings = FALSE)

# -- Data ----------------------------------------------------------------------
#
# Unit vectors around 2,000 topic directions with Zipf-like sizes and unequal
# spreads, so both values of k sit below the number of topics: the codebook
# regime, as for satellite or text embeddings.

make_codebook_data <- function(n, d, topics = 2000L) {
  dirs <- matrix(rnorm(topics * d), topics)
  dirs <- dirs / sqrt(rowSums(dirs^2))
  topic <- sample.int(topics, n, replace = TRUE, prob = 1 / seq_len(topics)^0.7)
  spread <- runif(topics, 0.02, 0.08)
  x <- dirs[topic, ] + matrix(rnorm(n * d), n) * spread[topic]
  x / sqrt(rowSums(x^2))
}

# Row-major doubles, so NumPy reads them with np.fromfile().reshape(-1, d).
write_rows <- function(x, path) writeBin(as.vector(t(x)), path, size = 8L)
read_rows <- function(path, d) {
  v <- readBin(path, "double", n = file.size(path) / 8)
  matrix(v, ncol = d, byrow = TRUE)
}

train_path <- "bench/data/emeans/train.f64"
test_path <- "bench/data/emeans/test.f64"
if (!file.exists(train_path) || !file.exists(test_path)) {
  set.seed(42)
  x_all <- make_codebook_data(n_train + n_test, d)
  write_rows(x_all[seq_len(n_train), ], train_path)
  write_rows(x_all[n_train + seq_len(n_test), ], test_path)
  rm(x_all)
}
x <- read_rows(train_path, d)

# -- Helpers -------------------------------------------------------------------

time_fit <- function(f, reps = 3L, slow = 30) {
  times <- numeric(0)
  repeat {
    gc(FALSE)
    t0 <- proc.time()[["elapsed"]]
    fit <- f()
    times <- c(times, proc.time()[["elapsed"]] - t0)
    if (length(times) == reps || times[1] > slow) break
  }
  list(seconds = stats::median(times), runs = length(times), fit = fit)
}

rows <- list()
record <- function(k, method, package, threads, timed, centroids) {
  file <- sprintf("bench/data/emeans/%s_k%d.f64", gsub("[^a-z0-9]+", "-", tolower(method)), k)
  write_rows(centroids, file)
  rows[[length(rows) + 1L]] <<- data.frame(
    k = k, method = method, package = package, threads = threads,
    seconds = timed$seconds, runs = timed$runs, centroids = basename(file)
  )
  cat(sprintf("  %-28s %8.2f s (%d run%s)\n", method, timed$seconds,
              timed$runs, if (timed$runs == 1L) "" else "s"))
}

threads <- shoal_threads()
have_clusterr <- requireNamespace("ClusterR", quietly = TRUE)
if (!have_clusterr) message("ClusterR is not installed; its mini-batch k-means is skipped.")

# -- Runs ----------------------------------------------------------------------

for (k in ks) {
  cat(sprintf("\n=== k = %d ===\n", k))

  em <- time_fit(function() shoal_emeans(x, k))
  record(k, "e-means", "shoal", threads, em, em$fit$centroids)
  batch <- em$fit$recipe$batch_size
  steps <- em$fit$recipe$total_steps

  km <- time_fit(function() shoal_kmeans(x, k, init = "random", n_runs = 1L, max_iter = lloyd_iters))
  record(k, "Lloyd", "shoal", threads, km, km$fit$centroids)
  if (k <= 256L) {
    pp <- time_fit(function() shoal_kmeans(x, k, n_runs = 1L, max_iter = lloyd_iters))
    record(k, "Lloyd, k-means++ start", "shoal", threads, pp, pp$fit$centroids)
  }

  # Single-threaded, and seeded through R's RNG. At this size it stops after
  # a few iterations, with the warning that its quick-transfer stage
  # exceeded its step limit (ifault 4); the score records where it stopped.
  hw <- time_fit(function() {
    set.seed(1L)
    suppressWarnings(stats::kmeans(x, k, iter.max = lloyd_iters))
  })
  record(k, "Hartigan-Wong", "stats", 1L, hw, hw$fit$centers)

  if (have_clusterr) {
    # The same budget as e-means: its batch size and number of updates, with
    # early stopping off (tol must be positive).
    mb <- time_fit(function() {
      ClusterR::MiniBatchKmeans(
        x, k, batch_size = batch, num_init = 1L, max_iters = steps,
        initializer = "random", early_stop_iter = steps, tol = 1e-12, seed = 1L
      )
    })
    record(k, "mini-batch", "ClusterR", 1L, mb, mb$fit$centroids)
  }
}

out <- do.call(rbind, rows)
write.csv(out, "bench/results_emeans_r.csv", row.names = FALSE)
cat("\nSaved: bench/results_emeans_r.csv\n")
