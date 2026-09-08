# shoal

shoal is a fast, foundational toolkit for clustering in R, with Rust
backends behind one consistent interface. Seven clustering algorithms,
from k-means to EVoC and graph SLIC, take a numeric matrix or data frame
and return one shared result class that prints and plots the same way,
with noise points as `NA`. Alongside them are the building blocks
clustering rests on: pairwise distance matrices and exact
nearest-neighbour search under nine metrics, and the indices for
choosing the number of clusters on evidence. Everything runs
multithreaded, and every function matches or improves on the best R and
Python alternatives.

| Function | Algorithm | Backend | Reach for it when |
|----|----|----|----|
| [`shoal_kmeans()`](https://belian-earth.github.io/shoal/reference/shoal_kmeans.md) | k-means | [linfa](https://github.com/rust-ml/linfa) | Clusters are roughly spherical and you know how many to expect. |
| [`shoal_gmm()`](https://belian-earth.github.io/shoal/reference/shoal_gmm.md) | Gaussian mixture | linfa | Clusters are elliptical, or you want soft memberships and [`BIC()`](https://rdrr.io/r/stats/AIC.html). |
| [`shoal_dbscan()`](https://belian-earth.github.io/shoal/reference/shoal_dbscan.md) | DBSCAN | [petal-clustering](https://github.com/petabi/petal-clustering) | Clusters have arbitrary shape and a common density. |
| [`shoal_hdbscan()`](https://belian-earth.github.io/shoal/reference/shoal_hdbscan.md) | HDBSCAN | petal-clustering | Clusters have arbitrary shape and varying density. |
| [`shoal_hclust()`](https://belian-earth.github.io/shoal/reference/shoal_hclust.md) | Agglomerative hierarchical | [kodama](https://github.com/diffeo/kodama) | You want a dendrogram and R’s [`cutree()`](https://rdrr.io/r/stats/cutree.html) ecosystem. |
| [`shoal_evoc()`](https://belian-earth.github.io/shoal/reference/shoal_evoc.md) | EVoC | In-tree port of [EVoC](https://github.com/TutteInstitute/evoc) | Rows are embedding vectors; you want every granularity at once. |
| [`shoal_slic()`](https://belian-earth.github.io/shoal/reference/shoal_slic.md) | Graph SLIC superpixels | In-tree | Points have coordinates and a neighbour graph; you want compact, homogeneous patches at a chosen scale. |

| Function | Does |
|----|----|
| [`shoal_dist()`](https://belian-earth.github.io/shoal/reference/shoal_dist.md) | Pairwise distances under nine metrics, returned as R’s own `dist` so [`cmdscale()`](https://rdrr.io/r/stats/cmdscale.html), [`cluster::pam()`](https://rdrr.io/pkg/cluster/man/pam.html) and the rest work without glue. |
| [`shoal_knn()`](https://belian-earth.github.io/shoal/reference/shoal_knn.md) | Exact k-nearest neighbours by kd-tree or parallel scan, same metrics, with a [`plot()`](https://rdrr.io/r/graphics/plot.default.html) that picks `eps` for DBSCAN. |
| [`shoal_silhouette()`](https://belian-earth.github.io/shoal/reference/shoal_silhouette.md), [`shoal_metrics()`](https://belian-earth.github.io/shoal/reference/shoal_metrics.md) | Silhouette widths, Calinski-Harabasz and Davies-Bouldin, for choosing the number of clusters. |
| [`shoal_energy()`](https://belian-earth.github.io/shoal/reference/shoal_energy.md), [`shoal_energy_self()`](https://belian-earth.github.io/shoal/reference/shoal_energy.md) | Two-sample energy distance between point clouds of any dimension, reduced in parallel without a distance matrix; the self term alone, for reuse across comparisons. |

## Installation

``` r

# install.packages("pak")
pak::pak("belian-earth/shoal")
```

Requires a working [Rust toolchain](https://rustup.rs/) (rustc \>=
1.81).

## A first look

The bundled `rings` data is three concentric rings plus scattered noise,
the case that defeats centroid methods. HDBSCAN finds the rings and
labels the noise without being told how many groups to look for.

``` r

library(shoal)

fit <- shoal_hdbscan(rings, min_cluster_size = 15L, min_samples = 5L)
fit
#> 
#> ── HDBSCAN Clustering
#> Metric: "euclidean"
#> Parameters: alpha = 1, min_samples = 5, min_cluster_size = 15, boruvka = TRUE
#> Clusters: 3, Noise points: 26
#> GLOSH outlier scores: median 0.079, max 0.965

plot(fit)
```

![](reference/figures/README-rings-1.png)

Every algorithm returns the same shape of object: `cluster` is the
assignment, `data` the matrix it was fitted to, and `params` records the
call. The same [`plot()`](https://rdrr.io/r/graphics/plot.default.html)
and [`print()`](https://rdrr.io/r/base/print.html) serve them all, so
swapping one algorithm for another is a one-word change.

The building blocks share the interface. Nearest neighbours of every
point, under any of the metrics
[`shoal_dist()`](https://belian-earth.github.io/shoal/reference/shoal_dist.md)
offers:

``` r

nn <- shoal_knn(rings, k = 5L, metric = "manhattan")
nn
#> 
#> ── k-Nearest Neighbours
#> Metric: "manhattan", Search: "kdtree"
#> Points: 1260, Neighbours: 5
#> Distance to neighbour 5: min 0.03437, median 0.1164, max 1.959

head(nn$id, 3)
#>        1   2   3   4   5
#> [1,]   2 165 236 183 283
#> [2,]   1 236 165 183 283
#> [3,] 265 233  59 224   9
```

## Learn more

- [`vignette("shoal")`](https://belian-earth.github.io/shoal/articles/shoal.md)
  introduces each algorithm in turn, with a picture of what it finds,
  then the distance and neighbour tools and how to choose the number of
  clusters.
- [`vignette("distances")`](https://belian-earth.github.io/shoal/articles/distances.md)
  is about the choices around
  [`shoal_dist()`](https://belian-earth.github.io/shoal/reference/shoal_dist.md)
  and
  [`shoal_knn()`](https://belian-earth.github.io/shoal/reference/shoal_knn.md):
  which metric, matrix or neighbours, tree or scan, and what to build
  from a neighbour result.
- [`vignette("umap")`](https://belian-earth.github.io/shoal/articles/umap.md)
  shows what embedding wide data with UMAP does to each algorithm.
- [`vignette("evoc")`](https://belian-earth.github.io/shoal/articles/evoc.md)
  clusters real sentence embeddings with EVoC and compares it with the
  alternatives.
- The [reference](https://belian-earth.github.io/shoal/reference/)
  documents every function.

## Performance

shoal is built for speed. Every clustering algorithm, the distance
matrix and the nearest-neighbour search are benchmarked against the best
R alternative and, where one exists, the Python one, at matched
settings. Each either matches or improves on them, by margins that grow
with the size and dimension of the data. The results, the scaling figure
and the scripts that produce them are in
[`bench/README.md`](https://github.com/belian-earth/shoal/blob/main/bench/README.md).

The Rust backends run their parallel stages on a thread pool owned by
the package, which by default uses every logical core.
`shoal_threads(n)` resizes it for the session;
`options(shoal.threads = n)` or `RAYON_NUM_THREADS` set the default
before the package loads. Results never depend on the thread count.

## Contributing

Build and workflow notes for working on the package, including the one
about timing, are in
[`DEVELOPMENT.md`](https://github.com/belian-earth/shoal/blob/main/DEVELOPMENT.md).

## Acknowledgements

The algorithms are the work of others. Density-based clustering is bound
to [petal-clustering](https://github.com/petabi/petal-clustering) by
[Petabi](https://github.com/petabi); hierarchical clustering to
[kodama](https://github.com/diffeo/kodama), a Rust port of
*fastcluster*; k-means and Gaussian mixtures to
[linfa](https://github.com/rust-ml/linfa); and EVoC to a Rust port of
the [reference implementation](https://github.com/TutteInstitute/evoc)
by Leland McInnes and the Tutte Institute, validated against fixtures
generated from it. shoal is an R interface to their work.
