# Reference relaxation in R, written independently of the Rust path: the
# slot-matrix form with Jacobi updates, then components and dust absorption
# with the sequential semantics the Rust code documents.
reference_slic <- function(x, xy, nb, init, lambda, s_nom, n_iter, min_patch = 1L,
                           adaptive = "none", alpha = 0) {
  patch <- as.integer(factor(init))
  n <- nrow(x); k <- ncol(nb)
  for (it in seq_len(n_iter)) {
    np <- max(patch)
    n_in <- tabulate(patch, np)
    pm <- rowsum(x, patch) / n_in
    pc <- rowsum(xy, patch) / n_in
    lam <- if (adaptive == "mean") {
      d2 <- rowSums((x - pm[patch, , drop = FALSE])^2)
      alpha * pmax(as.numeric(rowsum(d2, patch) / n_in), 1e-4)
    } else rep(lambda, np)
    nbp <- matrix(patch[nb], nrow = n)
    bix <- which(rowSums(nbp != patch, na.rm = TRUE) > 0)
    dist_to <- function(p) {
      ok <- !is.na(p); out <- rep(Inf, length(p))
      out[ok] <- rowSums((x[bix[ok], , drop = FALSE] - pm[p[ok], , drop = FALSE])^2) +
        lam[p[ok]] * rowSums((xy[bix[ok], , drop = FALSE] - pc[p[ok], , drop = FALSE])^2) / s_nom^2
      out
    }
    D <- matrix(NA_real_, length(bix), k + 1L)
    D[, 1L] <- dist_to(patch[bix])
    for (j in seq_len(k)) {
      p <- nbp[bix, j]; p[!is.na(p) & p == patch[bix]] <- NA
      D[, j + 1L] <- dist_to(p)
    }
    best <- max.col(-D, ties.method = "first")
    mv <- best > 1L
    patch[bix[mv]] <- nbp[cbind(bix[mv], best[mv] - 1L)]
  }
  # components by union over same-label edges, numbered by first point
  pairs <- do.call(rbind, lapply(seq_len(k), function(j) {
    ok <- !is.na(nb[, j]) & patch[nb[, j]] == patch
    cbind(seq_len(n)[ok], nb[ok, j])
  }))
  g <- igraph::graph_from_edgelist(pairs, directed = FALSE)
  g <- igraph::add_vertices(g, max(0, n - igraph::vcount(g)))
  patch <- as.integer(igraph::components(g)$membership)
  repeat {
    cnt <- tabulate(patch, max(patch)); dust <- which(cnt > 0 & cnt < min_patch)
    if (!length(dust)) break
    ix <- which(patch %in% dust); lab <- patch[ix]
    nbl <- matrix(patch[nb[ix, ]], nrow = length(ix)); moved <- FALSE
    for (p in dust) {
      sel <- which(lab == p); if (!length(sel)) next
      q <- nbl[sel, ]; q <- q[!is.na(q) & q != p]
      if (!length(q)) next
      lab[sel] <- as.integer(names(which.max(table(q)))); moved <- TRUE
    }
    if (!moved) break
    patch[ix] <- lab
  }
  as.integer(factor(patch))
}

same_partition <- function(a, b) {
  tb <- table(a, b)
  all(rowSums(tb > 0) == 1) && all(colSums(tb > 0) == 1)
}

grid_case <- function(side = 24L, seed = 1L) {
  set.seed(seed)
  g <- expand.grid(c = seq_len(side), r = seq_len(side))
  x <- cbind(as.numeric(g$c > side / 2), 0.2 * (g$r > side / 3)) +
    matrix(rnorm(2 * side^2, sd = 0.05), ncol = 2)
  xy <- cbind(g$c, g$r) * 10
  nn <- shoal_knn(xy, k = 4L)
  nb <- nn$id
  nb[nn$dist > 10] <- NA
  init <- as.integer(factor(paste((g$r - 1) %/% 4, (g$c - 1) %/% 4)))
  list(x = x, xy = xy, nb = nb, init = init, g = g)
}

test_that("shoal_slic matches the R reference exactly, fixed and adaptive", {
  skip_if_not_installed("igraph")
  cs <- grid_case()
  for (mp in c(1L, 6L)) {
    sp <- shoal_slic(cs$x, cs$xy, cs$nb, cs$init, lambda = 0.02, s_nom = 20, n_iter = 8L, min_patch = mp)
    ref <- reference_slic(cs$x, cs$xy, cs$nb, cs$init, 0.02, 20, 8L, min_patch = mp)
    expect_true(same_partition(sp$cluster, ref))
    expect_identical(sp$cluster, ref)
  }
  sp <- shoal_slic(cs$x, cs$xy, cs$nb, cs$init, adaptive = "mean", alpha = 0.3, s_nom = 20, n_iter = 8L)
  ref <- reference_slic(cs$x, cs$xy, cs$nb, cs$init, NA, 20, 8L, adaptive = "mean", alpha = 0.3)
  expect_identical(sp$cluster, ref)
})

test_that("patches follow the feature seams and the result is documented", {
  cs <- grid_case()
  sp <- shoal_slic(cs$x, cs$xy, cs$nb, cs$init, lambda = 0.02, s_nom = 20, n_iter = 10L, min_patch = 4L)
  expect_s3_class(sp, c("shoal_slic", "shoal_clustering"))
  expect_identical(sp$algorithm, "Graph SLIC")
  expect_identical(sp$n_noise, 0L)
  expect_length(sp$moves, 10L)
  expect_true(all(diff(sp$moves) <= 0) || sp$moves[10] < sp$moves[1])
  expect_length(sp$init_stat, max(cs$init))
  expect_length(sp$lambda_median, 10L)
  expect_true(all(sp$lambda_median == 0.02))
  # no patch straddles the vertical seam
  left <- cs$g$c <= 12
  expect_length(intersect(unique(sp$cluster[left]), unique(sp$cluster[!left])), 0L)
  expect_true(all(tabulate(sp$cluster) >= 4L))
  expect_identical(sort(unique(sp$cluster)), seq_len(sp$n_clusters))
})

test_that("init_stat is the per-tile mean squared distance to the tile mean", {
  cs <- grid_case()
  sp <- shoal_slic(cs$x, cs$xy, cs$nb, cs$init, lambda = 0.02, s_nom = 20, n_iter = 1L)
  init <- as.integer(factor(cs$init))
  pm <- rowsum(cs$x, init) / tabulate(init)
  d2 <- rowSums((cs$x - pm[init, ])^2)
  expect_equal(sp$init_stat, as.numeric(rowsum(d2, init) / tabulate(init)))
})

test_that("early stopping and input validation behave", {
  cs <- grid_case()
  sp <- shoal_slic(cs$x, cs$xy, cs$nb, cs$init, lambda = 0.02, s_nom = 20, n_iter = 50L, tol = 0.01)
  expect_lt(length(sp$moves), 50L)
  expect_error(shoal_slic(cs$x, cs$xy[-1, ], cs$nb, cs$init, lambda = 1), "rows and 2 columns")
  expect_error(shoal_slic(cs$x, cs$xy, cs$nb[-1, ], cs$init, lambda = 1), "one row per point")
  bad <- cs$nb; bad[1, 1] <- 0L
  expect_error(shoal_slic(cs$x, cs$xy, bad, cs$init, lambda = 1), "outside")
  expect_error(shoal_slic(cs$x, cs$xy, cs$nb, cs$init[-1], lambda = 1), "length")
  expect_error(shoal_slic(cs$x, cs$xy, cs$nb, cs$init, adaptive = "mean"), "alpha")
  expect_error(shoal_slic(cs$x, cs$xy, cs$nb, cs$init, lambda = 1, tol = 1), "tol")
})

test_that("print reports iterations and variability", {
  cs <- grid_case()
  sp <- shoal_slic(cs$x, cs$xy, cs$nb, cs$init, lambda = 0.02, s_nom = 20, n_iter = 3L)
  out <- paste(cli::cli_fmt(print(sp)), collapse = "\n")
  expect_match(out, "Graph SLIC Clustering")
  expect_match(out, "Iterations run: 3")
  expect_match(out, "variability")
})
