# Graph SLIC Superpixels

Partitions a point set into compact, feature-homogeneous patches by the
SLIC relaxation (Achanta et al. 2012) run on an arbitrary neighbour
graph rather than a raster. Starting from initial labels, every boundary
point is moved, all points at once, to the adjacent patch whose feature
mean is closest, with a spatial term that keeps patches compact:

## Usage

``` r
shoal_slic(
  x,
  xy,
  nb,
  init,
  lambda = 1,
  lambda_scale = NULL,
  s_nom = 1,
  n_iter = 25L,
  adaptive = c("none", "mean"),
  alpha = NULL,
  min_patch = 1L,
  tol = 0
)
```

## Arguments

- x:

  A numeric matrix or data frame of features, one row per point and any
  number of columns, one included. Data frames are coerced to a matrix
  using their numeric columns. Rows with missing or non-finite values
  are an error.

- xy:

  A numeric matrix with two columns of coordinates, one row per point,
  in the units `s_nom` is expressed in.

- nb:

  An integer matrix with one row per point whose columns hold the row
  indices of that point's neighbours, `NA` for an unused slot, as
  `shoal_knn()$id` or `dbscan::kNN()$id` give after distant neighbours
  are set to `NA`. Need not be symmetric.

- init:

  An integer vector of initial patch labels, one per point. Any
  labelling works; a regular tiling at the intended patch scale is the
  usual choice.

- lambda:

  Spatial weight at `s_nom`. Ignored when `lambda_scale` is given or
  `adaptive` is not `"none"`.

- lambda_scale:

  Optional. Sets the spatial weight to this multiple of the median of
  `init_stat`, so one setting carries across data of different feature
  scales. Ignored when `adaptive` is not `"none"`.

- s_nom:

  Nominal radius, in the units of `xy`. Default `1`.

- n_iter:

  Iterations. Default `25L`. The `moves` component shows whether that
  was enough: the count typically decays geometrically.

- adaptive:

  `"none"` (default) for one global `lambda`, or `"mean"` for per-patch
  weights of `alpha` times the patch's mean squared feature distance.

- alpha:

  Multiplier for `adaptive = "mean"`; required then, ignored otherwise.

- min_patch:

  Smallest patch kept after relaxation, in points. Default `1L` keeps
  every component.

- tol:

  Stop early once fewer than this share of points move in an iteration.
  Default `0` runs all `n_iter`.

## Value

An object of class `c("shoal_slic", "shoal_clustering")`: a list with
components `cluster` (integer patch labels), `n_clusters`, `n_noise`
(always `0`), `data`, `algorithm`, `params` (with `lambda` the weight
actually used, resolved from `lambda_scale` where that was given),
`moves` (points moved in each iteration run), `init_stat` (mean squared
feature distance of each initial patch's points to its mean, named by
the values of `init`) and `lambda_median` (median per-patch weight over
non-empty patches in each iteration run).

## Details

\$\$\\x_i - m_p\\^2 + \lambda_p \\ \\xy_i - c_p\\^2 / s\_{nom}^2\$\$

where \\m_p\\ and \\c_p\\ are the patch's feature and coordinate means
at the start of the iteration. Because the adjacency and the seeding are
inputs, the same routine serves a raster, a hexagonal or pentagonal
discrete global grid, or any point cloud with a neighbour structure.

## Compactness

`lambda` weights the spatial term at the nominal radius `s_nom`, in the
units of `xy`: a point `s_nom` from its patch centre pays `lambda` in
the squared-feature-distance currency of `x`. The scale of that currency
is data dependent, so `init_stat` in the result gives the mean squared
feature distance of each initial patch's points to its mean, the local
variability statistic of `supercells::sc_tune_compactness()`. Expressing
`lambda` as a multiple of its median carries a tuned setting to other
data, and `lambda_scale` does exactly that: the weight used is
`lambda_scale` times the median of `init_stat`, and is reported as
`lambda` in the result. Balancing the two terms one-for-one
(`lambda_scale = 1`) is a common default that over-regularises
high-dimensional embeddings; on those, values of a tenth or less are
what track the data.

`adaptive = "mean"` rescales the spatial term per patch to `alpha` times
that patch's own mean squared feature distance at the start of each
iteration, in the manner of ASLIC: the balance is then relative to local
contrast, so homogeneous and heterogeneous regions are regularised alike
in relative terms. The per-patch statistic is floored at `1e-4` so a
perfectly uniform patch keeps a spatial term.

## Post-processing

After the iterations each patch is split into its connected components
over `nb`, and patches of fewer than `min_patch` points are absorbed
into the neighbouring patch they share the most edges with, until none
remain. Labels are then renumbered `1..n_clusters`.

Every step is deterministic. Ties in the move decision go to the point's
own patch, then to the earlier column of `nb`; component ids follow
first appearance in row order.

## References

Achanta R, Shaji A, Smith K, Lucchi A, Fua P, Süsstrunk S (2012). SLIC
superpixels compared to state-of-the-art superpixel methods. IEEE
Transactions on Pattern Analysis and Machine Intelligence 34(11),
2274-2282.

## Examples

``` r
# A 20 x 20 grid with two feature regions and a 4-neighbour graph.
side <- 20L
g <- expand.grid(c = seq_len(side), r = seq_len(side))
x <- cbind(as.numeric(g$c > side / 2), 0.1 * g$r / side) + rnorm(2 * side^2, sd = 0.02)
xy <- cbind(g$c, g$r) * 10
nn <- shoal_knn(xy, k = 4L)
nb <- nn$id
nb[nn$dist > 10] <- NA
init <- as.integer(factor(paste((g$r - 1) %/% 4, (g$c - 1) %/% 4)))
sp <- shoal_slic(x, xy, nb, init, lambda = 0.01, s_nom = 20, n_iter = 10L)
sp
#> 
#> ── Graph SLIC Clustering 
#> Parameters: lambda = 0.01, lambda_scale = NA, s_nom = 20, n_iter = 10, adaptive
#> = none, alpha = NA, min_patch = 1
#> Clusters: 20, Noise points: 0
#> Iterations run: 10; points moved: first 40, last 0
#> Initial-patch variability (median): 0.0008034
table(sp$cluster, x[, 1] > 0.5)[1:5, ]
#>    
#>     FALSE TRUE
#>   1    20    0
#>   2    20    0
#>   3     0   20
#>   4     0   20
#>   5    20    0

# The same weight expressed relative to the data's own variability.
sp2 <- shoal_slic(x, xy, nb, init, lambda_scale = 0.1, s_nom = 20, n_iter = 10L)
sp2$params$lambda
#> [1] 8.034354e-05
```
