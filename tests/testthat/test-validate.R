test_that("check_numeric_matrix accepts numeric matrices", {
  m <- matrix(1:6, nrow = 3, ncol = 2)
  expect_equal(check_numeric_matrix(m), m)
})

test_that("check_numeric_matrix coerces data frames", {
  df <- data.frame(a = 1:5, b = 6:10, c = letters[1:5])
  result <- check_numeric_matrix(df)

  expect_true(is.matrix(result))
  expect_true(is.numeric(result))
  expect_equal(ncol(result), 2L)  # drops character column
  expect_equal(nrow(result), 5L)
})

test_that("check_numeric_matrix drops non-numeric columns from data frames", {
  df <- data.frame(
    x = 1:5, y = 6:10, z = seq(0.1, 0.5, by = 0.1),
    label = letters[1:5],
    flag = c(TRUE, FALSE, TRUE, FALSE, TRUE)
  )
  result <- check_numeric_matrix(df)
  # keeps x, y, z (numeric); drops label (character) and flag (logical)
  expect_equal(ncol(result), 3L)
})

test_that("check_numeric_matrix rejects non-matrix non-data-frame", {
  expect_error(check_numeric_matrix(1:10), "numeric matrix or data frame")
  expect_error(check_numeric_matrix("abc"), "numeric matrix or data frame")
  expect_error(check_numeric_matrix(list(1, 2)), "numeric matrix or data frame")
})

test_that("check_numeric_matrix rejects data frame with < 2 numeric columns", {
  df <- data.frame(a = 1:5, b = letters[1:5])
  expect_error(check_numeric_matrix(df), "at least 2 numeric columns")

  df2 <- data.frame(a = letters[1:5], b = letters[6:10])
  expect_error(check_numeric_matrix(df2), "at least 2 numeric columns")
})

test_that("check_numeric_matrix rejects matrix with < 2 columns", {
  m <- matrix(1:5, ncol = 1)
  expect_error(check_numeric_matrix(m), "at least 2 columns")
})

test_that("check_numeric_matrix drops rows with NAs and warns", {
  m <- matrix(c(1, 2, NA, 4, 5, 6, 7, 8), nrow = 4, ncol = 2)
  expect_warning(result <- check_numeric_matrix(m), "Removed 1 row")
  expect_equal(nrow(result), 3L)
  expect_false(anyNA(result))
})

test_that("check_numeric_matrix errors when all rows have NAs", {
  m <- matrix(c(NA, NA, 1, 2), nrow = 2, ncol = 2)
  m[2, 2] <- NA
  expect_error(
    suppressWarnings(check_numeric_matrix(m)),
    "No complete rows"
  )
})

test_that("check_numeric_matrix passes clean data without warning", {
  m <- matrix(1:6, nrow = 3, ncol = 2)
  expect_no_warning(check_numeric_matrix(m))
})

test_that("check_positive_number validates correctly", {
  expect_invisible(check_positive_number(1.0))
  expect_invisible(check_positive_number(0.001))

  expect_error(check_positive_number(-1), "positive")
  expect_error(check_positive_number(0), "positive")
  expect_error(check_positive_number(NA_real_), "positive")
  expect_error(check_positive_number("a"), "single number")
  expect_error(check_positive_number(1:2), "single number")
})

test_that("check_non_negative_number validates correctly", {
  expect_invisible(check_non_negative_number(0))
  expect_invisible(check_non_negative_number(0.5))
  expect_invisible(check_non_negative_number(3L))

  expect_error(check_non_negative_number(-0.1), "non-negative")
  expect_error(check_non_negative_number(Inf), "non-negative")
  expect_error(check_non_negative_number(NA_real_), "non-negative")
  expect_error(check_non_negative_number("a"), "single number")
  expect_error(check_non_negative_number(1:2), "single number")
})

test_that("check_positive_integer validates correctly", {
  expect_invisible(check_positive_integer(1L))
  expect_invisible(check_positive_integer(5L))
  expect_invisible(check_positive_integer(100))  # integerish double OK

  expect_error(check_positive_integer(0L), "positive integer")
  expect_error(check_positive_integer(-1L), "positive integer")
  expect_error(check_positive_integer(NA_integer_), "positive integer")
  expect_error(check_positive_integer(1.5), "single integer")
  expect_error(check_positive_integer("a"), "single integer")
})
