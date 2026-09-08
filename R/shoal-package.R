#' @title High-Performance Clustering via Rust
#'
#' @description
#' A small, deliberately foundational set of clustering algorithms with Rust
#' backends, presented behind one consistent interface. Every result carries the
#' shared `"shoal_clustering"` class, so `print()` and `plot()` behave the same
#' way whichever algorithm produced it, and noise points are `NA` in the
#' cluster vector wherever an algorithm has them.
#'
#' # Algorithms
#'
#' - [shoal_kmeans()]: k-means, partitioning into a fixed number of compact
#'   clusters, with `predict()` for new observations.
#' - [shoal_gmm()]: Gaussian mixtures, soft probabilistic assignment with
#'   elliptical clusters, and `logLik()` so `AIC()` and `BIC()` can choose the
#'   component count.
#' - [shoal_dbscan()]: DBSCAN, clusters as dense regions of one density
#'   separated by sparser space.
#' - [shoal_hdbscan()]: HDBSCAN, the hierarchical extension that adapts to
#'   clusters of varying density and scores outliers.
#' - [shoal_hclust()]: agglomerative hierarchical clustering with seven
#'   linkage methods, returning a standard [stats::hclust] object.
#' - [shoal_evoc()]: EVoC, direct multi-granularity clustering of embedding
#'   vectors, returning every cluster layer rather than one flat partition.
#' - [shoal_slic()]: graph SLIC superpixels, compact feature-homogeneous
#'   patches over any point set with coordinates and a neighbour graph.
#'
#' # Supporting functions
#'
#' - [shoal_dist()]: pairwise distance matrices, returning a standard
#'   [stats::dist] object.
#' - [shoal_knn()]: exact k-nearest neighbours, also the neighbour graph
#'   [shoal_slic()] runs on.
#' - [shoal_silhouette()] and [shoal_metrics()]: validity measures for
#'   choosing the number of clusters.
#' - [shoal_energy()] and [shoal_energy_self()]: the two-sample energy
#'   distance between point clouds, for comparing distributions without
#'   clustering them.
#' - [shoal_palette()]: the default cluster colours used by `plot()`.
#' - [shoal_threads()]: the size of the thread pool the Rust backends use.
#'
#' # Data and vignettes
#'
#' [rings] and [newsgroups] are bundled examples: concentric rings with noise
#' for the density algorithms, and real sentence embeddings for EVoC.
#' `vignette("shoal")` introduces each algorithm with a picture of what it
#' finds, `vignette("umap")` shows what a UMAP embedding does to each of them
#' on wide tabular data, and `vignette("evoc")` clusters the sentence
#' embeddings and compares EVoC with the alternatives.
#'
#' # Design
#'
#' k-means, the mixture and the density-based algorithms take numeric matrices
#' or data frames directly. Hierarchical clustering works from a distance
#' matrix, so it accepts anything [stats::dist()] would produce as well as raw
#' data. Returning R's own `dist` and `hclust` classes rather than bespoke ones
#' is deliberate: `cutree()`, `as.dendrogram()`, `cmdscale()` and the rest of
#' the ecosystem then work on the results without any glue.
#'
#' The density-based algorithms are bindings to the
#' \href{https://github.com/petabi/petal-clustering}{petal-clustering} Rust
#' crate by \href{https://github.com/petabi}{Petabi, Inc.}; hierarchical
#' clustering is bound to \href{https://github.com/diffeo/kodama}{kodama}, a
#' Rust port of \emph{fastcluster}; k-means and Gaussian mixtures to
#' \href{https://github.com/rust-ml/linfa}{linfa}; and EVoC to an in-tree
#' Rust port of the
#' \href{https://github.com/TutteInstitute/evoc}{reference implementation},
#' validated against fixtures generated from it.
#'
#' @keywords internal
#' @aliases shoal-package
"_PACKAGE"

#' @useDynLib shoal, .registration = TRUE
#' @importFrom rlang %||%
NULL
