# K-Means Clustering

Lloyd's algorithm via the Rust [linfa](https://github.com/rust-ml/linfa)
toolkit, with k-means++ and k-means\|\| initialisation.

## Usage

``` r
shoal_kmeans(
  x,
  k,
  init = c("kmeans++", "kmeans_parallel", "random"),
  n_runs = 10L,
  max_iter = 300L,
  tolerance = 1e-04,
  seed = 1L
)
```

## Arguments

- x:

  A numeric matrix or data frame. Data frames are coerced to a matrix
  using their numeric columns (non-numeric columns are dropped).

- k:

  Number of clusters. Required, since there is no sensible default for
  the central modelling decision. Must be at least 1 and no more than
  `nrow(x)`.

- init:

  Initialisation method. `"kmeans++"` (default) is the usual choice;
  `"kmeans_parallel"` scales better past roughly 100 clusters;
  `"random"` is the naive baseline.

- n_runs:

  Number of restarts, keeping the fit with the lowest inertia. Default
  `10L`.

- max_iter:

  Maximum iterations per run. Default `300L`.

- tolerance:

  Convergence threshold on centroid movement. Default `1e-4`.

- seed:

  Non-negative whole-number seed for initialisation. Stored and passed
  as a double, so values beyond the integer range are safe. Default
  `1L`.

## Value

An object of class `c("shoal_kmeans", "shoal_clustering")`: a list with
components `cluster` (integer vector of cluster IDs), `n_clusters`,
`n_noise` (always `0`), `data`, `algorithm`, `params`, `centroids` (a
`k x ncol(x)` matrix), `inertia` and `sizes`.

## Details

Unlike the density-based algorithms, k-means partitions every
observation: there is no noise class, so `cluster` never contains `NA`.
It is also predictive:
[`predict()`](https://rdrr.io/r/stats/predict.html) assigns new
observations to the fitted centroids.

## Reproducibility

k-means is stochastic in its initialisation, so `seed` is a parameter
rather than being taken from R's RNG. The same `seed` and parameters
always give the same partition;
[`set.seed()`](https://rdrr.io/r/base/Random.html) has no effect on it.

## Choosing k

`inertia` (the within-cluster sum of squares) is returned so it can be
compared across values of `k`, the usual scree or elbow approach. Note
that it decreases monotonically with `k` by construction, so it
identifies a diminishing-returns point rather than an optimum.

## See also

[`predict.shoal_kmeans()`](https://belian-earth.github.io/shoal/reference/predict.shoal_kmeans.md)
for assigning new observations, and
[`shoal_emeans()`](https://belian-earth.github.io/shoal/reference/shoal_emeans.md)
for many clusters on large data.

## Examples

``` r
fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L)
fit
#> 
#> ── K-Means Clustering 
#> Parameters: k = 3, init = kmeans++, n_runs = 10, seed = 1
#> Clusters: 3, Noise points: 0
#> Within-cluster sum of squares: 78.851
#> Cluster sizes: 50, 38, 62
fit$centroids
#>      Sepal.Length Sepal.Width Petal.Length Petal.Width
#> [1,]     5.006000    3.428000     1.462000    0.246000
#> [2,]     6.850000    3.073684     5.742105    2.071053
#> [3,]     5.901613    2.748387     4.393548    1.433871
```
