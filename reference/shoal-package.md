# High-Performance Clustering via Rust

A small, deliberately foundational set of clustering algorithms with
Rust backends, presented behind one consistent interface. Every result
carries the shared `"shoal_clustering"` class, so
[`print()`](https://rdrr.io/r/base/print.html) and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) behave the same
way whichever algorithm produced it, and noise points are `NA` in the
cluster vector wherever an algorithm has them.

## Algorithms

- [`shoal_kmeans()`](https://belian-earth.github.io/shoal/reference/shoal_kmeans.md):
  k-means, partitioning into a fixed number of compact clusters, with
  [`predict()`](https://rdrr.io/r/stats/predict.html) for new
  observations.

- [`shoal_gmm()`](https://belian-earth.github.io/shoal/reference/shoal_gmm.md):
  Gaussian mixtures, soft probabilistic assignment with elliptical
  clusters, and [`logLik()`](https://rdrr.io/r/stats/logLik.html) so
  [`AIC()`](https://rdrr.io/r/stats/AIC.html) and
  [`BIC()`](https://rdrr.io/r/stats/AIC.html) can choose the component
  count.

- [`shoal_dbscan()`](https://belian-earth.github.io/shoal/reference/shoal_dbscan.md):
  DBSCAN, clusters as dense regions of one density separated by sparser
  space.

- [`shoal_hdbscan()`](https://belian-earth.github.io/shoal/reference/shoal_hdbscan.md):
  HDBSCAN, the hierarchical extension that adapts to clusters of varying
  density and scores outliers.

- [`shoal_hclust()`](https://belian-earth.github.io/shoal/reference/shoal_hclust.md):
  agglomerative hierarchical clustering with seven linkage methods,
  returning a standard
  [stats::hclust](https://rdrr.io/r/stats/hclust.html) object.

- [`shoal_evoc()`](https://belian-earth.github.io/shoal/reference/shoal_evoc.md):
  EVoC, direct multi-granularity clustering of embedding vectors,
  returning every cluster layer rather than one flat partition.

- [`shoal_slic()`](https://belian-earth.github.io/shoal/reference/shoal_slic.md):
  graph SLIC superpixels, compact feature-homogeneous patches over any
  point set with coordinates and a neighbour graph.

## Supporting functions

- [`shoal_dist()`](https://belian-earth.github.io/shoal/reference/shoal_dist.md):
  pairwise distance matrices, returning a standard
  [stats::dist](https://rdrr.io/r/stats/dist.html) object.

- [`shoal_knn()`](https://belian-earth.github.io/shoal/reference/shoal_knn.md):
  exact k-nearest neighbours, also the neighbour graph
  [`shoal_slic()`](https://belian-earth.github.io/shoal/reference/shoal_slic.md)
  runs on.

- [`shoal_silhouette()`](https://belian-earth.github.io/shoal/reference/shoal_silhouette.md)
  and
  [`shoal_metrics()`](https://belian-earth.github.io/shoal/reference/shoal_metrics.md):
  validity measures for choosing the number of clusters.

- [`shoal_energy()`](https://belian-earth.github.io/shoal/reference/shoal_energy.md)
  and
  [`shoal_energy_self()`](https://belian-earth.github.io/shoal/reference/shoal_energy.md):
  the two-sample energy distance between point clouds, for comparing
  distributions without clustering them.

- [`shoal_palette()`](https://belian-earth.github.io/shoal/reference/shoal_palette.md):
  the default cluster colours used by
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html).

- [`shoal_threads()`](https://belian-earth.github.io/shoal/reference/shoal_threads.md):
  the size of the thread pool the Rust backends use.

## Data and vignettes

[rings](https://belian-earth.github.io/shoal/reference/rings.md) and
[newsgroups](https://belian-earth.github.io/shoal/reference/newsgroups.md)
are bundled examples: concentric rings with noise for the density
algorithms, and real sentence embeddings for EVoC.
[`vignette("shoal")`](https://belian-earth.github.io/shoal/articles/shoal.md)
introduces each algorithm with a picture of what it finds,
[`vignette("umap")`](https://belian-earth.github.io/shoal/articles/umap.md)
shows what a UMAP embedding does to each of them on wide tabular data,
and
[`vignette("evoc")`](https://belian-earth.github.io/shoal/articles/evoc.md)
clusters the sentence embeddings and compares EVoC with the
alternatives.

## Design

k-means, the mixture and the density-based algorithms take numeric
matrices or data frames directly. Hierarchical clustering works from a
distance matrix, so it accepts anything
[`stats::dist()`](https://rdrr.io/r/stats/dist.html) would produce as
well as raw data. Returning R's own `dist` and `hclust` classes rather
than bespoke ones is deliberate:
[`cutree()`](https://rdrr.io/r/stats/cutree.html),
[`as.dendrogram()`](https://rdrr.io/r/stats/dendrogram.html),
[`cmdscale()`](https://rdrr.io/r/stats/cmdscale.html) and the rest of
the ecosystem then work on the results without any glue.

The density-based algorithms are bindings to the
[petal-clustering](https://github.com/petabi/petal-clustering) Rust
crate by [Petabi, Inc.](https://github.com/petabi); hierarchical
clustering is bound to [kodama](https://github.com/diffeo/kodama), a
Rust port of *fastcluster*; k-means and Gaussian mixtures to
[linfa](https://github.com/rust-ml/linfa); and EVoC to an in-tree Rust
port of the [reference
implementation](https://github.com/TutteInstitute/evoc), validated
against fixtures generated from it.

## See also

Useful links:

- <https://belian-earth.github.io/shoal/>

- <https://github.com/belian-earth/shoal>

- Report bugs at <https://github.com/belian-earth/shoal/issues>

## Author

**Maintainer**: First Last <first.last@example.com>

Authors:

- First Last <first.last@example.com>
