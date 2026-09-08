# Two-Sample Energy Distance

The energy distance of Szekely and Rizzo between the rows of `x` and the
rows of `y`, treated as two samples:

## Usage

``` r
shoal_energy(x, y, terms = FALSE, self_x = NULL, self_y = NULL)

shoal_energy_self(x)
```

## Arguments

- x, y:

  Numeric matrices or data frames with the same number of columns, one
  included. Data frames are coerced to matrices using their numeric
  columns. Rows with missing or non-finite values are an error.

- terms:

  Also return the three mean distances. Default `FALSE`.

- self_x, self_y:

  Optional precomputed within-sample terms from `shoal_energy_self()`,
  used in place of computing them.

  The cross term averages over all `n m` pairs, including the zero
  distances between a row and its own copy when `x` and `y` share rows,
  so a sample compared with itself comes out slightly below zero before
  the floor at zero is applied.

## Value

`shoal_energy()`: the energy distance, a single non-negative number;
with `terms = TRUE`, a list with `energy`, `cross` (mean cross-pair
distance), `self_x` and `self_y` (mean within-sample distances).
`shoal_energy_self()`: the mean distance over all distinct pairs of rows
of `x`, `0` for a single row.

## Details

\$\$E(X, Y) = 2\\\mathrm{E}\\x - y\\ - \mathrm{E}\\x - x'\\ -
\mathrm{E}\\y - y'\\\$\$

estimated by the mean Euclidean distance over all cross pairs and over
all distinct within-sample pairs. It is zero exactly when the two
distributions agree, positive otherwise, and needs no clustering,
density estimate or choice of bins, so it compares point clouds of any
dimension directly: two embeddings of the same region in different
years, a sample against a reference population, or a cluster against the
rest.

The three pairwise-distance blocks are reduced row by row in parallel in
Rust and never held in memory, so the cost is about
`(2 n m + n^2 + m^2) d / 2` flops and the memory is the two inputs. When
one sample is compared against many others its within-sample term is the
same every time: `shoal_energy_self()` computes that term alone, and
passing it as `self_x` or `self_y` skips recomputing it. For
unit-normalised rows the Euclidean distance is a monotone function of
cosine distance, so the statistic can be used on embeddings that are
compared by cosine.

## References

Szekely GJ, Rizzo ML (2013). Energy statistics: A class of statistics
based on distances. Journal of Statistical Planning and Inference
143(8), 1249-1272.

## Examples

``` r
x <- as.matrix(iris[iris$Species == "setosa", 1:4])
y <- as.matrix(iris[iris$Species == "versicolor", 1:4])
z <- as.matrix(iris[iris$Species == "virginica", 1:4])
shoal_energy(x, y)
#> [1] 4.908269
shoal_energy(y, z)  # closer pair
#> [1] 1.510683
shoal_energy(x, x, terms = TRUE)
#> $energy
#> [1] 0
#> 
#> $cross
#> [1] 0.6828805
#> 
#> $self_x
#> [1] 0.6968169
#> 
#> $self_y
#> [1] 0.6968169
#> 

# One reference against several samples: compute its own term once.
self_x <- shoal_energy_self(x)
shoal_energy(x, y, self_x = self_x)
#> [1] 4.908269
shoal_energy(x, z, self_x = self_x)
#> [1] 7.774686
```
