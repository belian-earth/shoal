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

test_that("energy is symmetric, zero-floored and orders shifted samples", {
  set.seed(4)
  x <- matrix(rnorm(50 * 3), ncol = 3)
  expect_equal(shoal_energy(x, x + 1), shoal_energy(x + 1, x))
  expect_gte(shoal_energy(x, x), 0)
  expect_lt(shoal_energy(x, x + 0.2), shoal_energy(x, x + 1))
  expect_equal(shoal_energy(x, x[sample(nrow(x)), ]), 0)
})

test_that("shoal_energy validates its inputs", {
  x <- matrix(rnorm(20), ncol = 2)
  expect_error(shoal_energy(x, matrix(rnorm(30), ncol = 3)), "same number of columns")
  x[1, 1] <- NA
  expect_error(shoal_energy(x, x), "missing")
})
