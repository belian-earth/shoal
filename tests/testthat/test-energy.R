reference_energy <- function(x, y) {
  d <- function(a, b) as.matrix(stats::dist(rbind(a, b)))[seq_len(nrow(a)), nrow(a) + seq_len(nrow(b))]
  self <- function(a) { m <- as.matrix(stats::dist(a)); sum(m) / (nrow(a) * (nrow(a) - 1)) }
  list(energy = 2 * mean(d(x, y)) - self(x) - self(y), cross = mean(d(x, y)), self_x = self(x), self_y = self(y))
}

test_that("shoal_energy matches the R reference", {
  set.seed(3)
  x <- matrix(rnorm(80 * 5), ncol = 5)
  y <- matrix(rnorm(60 * 5, mean = 0.3), ncol = 5)
  ref <- reference_energy(x, y)
  got <- shoal_energy(x, y, terms = TRUE)
  expect_equal(got$energy, ref$energy)
  expect_equal(got$cross, ref$cross)
  expect_equal(got$self_x, ref$self_x)
  expect_equal(got$self_y, ref$self_y)
  expect_equal(shoal_energy(x, y), ref$energy)
})

test_that("tiled products agree with the reference across tile boundaries", {
  # Samples larger than one row tile of the Rust kernel, unit-norm as for
  # embeddings compared by cosine, and far from the origin where the norm
  # expansion cancels most.
  set.seed(7)
  unit <- function(n, d) {
    m <- matrix(rnorm(n * d), n, d)
    m / sqrt(rowSums(m^2))
  }
  x <- unit(300, 8)
  y <- unit(250, 8)
  ref <- reference_energy(x, y)
  got <- shoal_energy(x, y, terms = TRUE)
  expect_equal(got, ref, tolerance = 1e-9)
  expect_equal(shoal_energy_self(y), ref$self_y, tolerance = 1e-9)
  far <- matrix(rnorm(200 * 6, mean = 100), ncol = 6)
  ref <- reference_energy(far, far + 0.5)
  expect_equal(shoal_energy(far, far + 0.5, terms = TRUE), ref, tolerance = 1e-9)
})

test_that("samples wider than one column tile take the multi-tile path", {
  # The Rust kernel forms the product in tiles of 2048 columns; a second
  # sample past that width exercises the tile boundary in the cross term
  # and, through its self term, in the upper triangle too.
  set.seed(8)
  x <- matrix(rnorm(60 * 4), ncol = 4)
  y <- matrix(rnorm(2100 * 4, mean = 0.2), ncol = 4)
  ref <- reference_energy(x, y)
  expect_equal(shoal_energy(x, y, terms = TRUE), ref, tolerance = 1e-9)
  expect_equal(shoal_energy_self(y), ref$self_y, tolerance = 1e-9)
  expect_equal(shoal_energy(y, x), ref$energy, tolerance = 1e-9)
})

test_that("energy is symmetric, zero-floored and orders shifted samples", {
  set.seed(4)
  x <- matrix(rnorm(50 * 3), ncol = 3)
  expect_equal(shoal_energy(x, x + 1), shoal_energy(x + 1, x))
  expect_gte(shoal_energy(x, x), 0)
  expect_lt(shoal_energy(x, x + 0.2), shoal_energy(x, x + 1))
  expect_equal(shoal_energy(x, x[sample(nrow(x)), ]), 0)
})

test_that("precomputed self terms are used and match", {
  set.seed(5)
  x <- matrix(rnorm(40 * 4), ncol = 4)
  y <- matrix(rnorm(30 * 4, mean = 0.5), ncol = 4)
  full <- shoal_energy(x, y, terms = TRUE)
  sx <- shoal_energy_self(x)
  expect_equal(sx, full$self_x)
  expect_equal(shoal_energy(x, y, self_x = sx), full$energy)
  got <- shoal_energy(x, y, terms = TRUE, self_x = sx, self_y = full$self_y)
  expect_equal(got, full)
  # A supplied term is used as given, not recomputed.
  expect_equal(shoal_energy(x, y, self_x = 0, terms = TRUE)$self_x, 0)
  expect_identical(shoal_energy_self(x[1, , drop = FALSE]), 0)
  expect_error(shoal_energy(x, y, self_x = -1), "non-negative")
})

test_that("one-column samples are accepted", {
  set.seed(6)
  x <- matrix(rnorm(50), ncol = 1)
  y <- matrix(rnorm(40, mean = 2), ncol = 1)
  ref <- reference_energy(x, y)
  expect_equal(shoal_energy(x, y), ref$energy)
  expect_equal(shoal_energy_self(x), ref$self_x)
})

test_that("shoal_energy validates its inputs", {
  x <- matrix(rnorm(20), ncol = 2)
  expect_error(shoal_energy(x, matrix(rnorm(30), ncol = 3)), "same number of columns")
  x[1, 1] <- NA
  expect_error(shoal_energy(x, x), "missing")
})
