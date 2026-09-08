use extendr_api::prelude::*;
use extendr_api::{throw_r_error, unwrap_or_throw_error};
use petal_clustering::{Dbscan, Fit, HDbscan};
use petal_neighbors::distance::{Cosine, Euclidean};

mod convert;
mod dist;
mod energy;
mod evoc;
mod gmm;
mod hclust;
mod kdtree;
mod kmeans;
mod knn;
mod metrics;
mod slic;
mod threads;
use convert::{
    assignment_vector, partial_labels_from_list, rmatrix_to_array2, rmatrix_to_array2_linfa,
};

#[extendr]
fn rust_dbscan(x: RMatrix<f64>, eps: f64, min_samples: i32, metric: &str) -> Integers {
    let data = rmatrix_to_array2(x);
    let n_points = data.nrows();
    let min_samples = min_samples as usize;

    let clusters = threads::pool().install(|| match metric {
        "euclidean" => {
            let mut model = Dbscan::new(eps, min_samples, Euclidean::default());
            model.fit(&data, None).0
        }
        "cosine" => {
            let mut model = Dbscan::new(eps, min_samples, Cosine::default());
            model.fit(&data, None).0
        }
        _ => panic!("Unknown metric: {metric}"),
    });

    Integers::from_values(assignment_vector(&clusters, n_points))
}

#[extendr]
fn rust_hdbscan(
    x: RMatrix<f64>,
    alpha: f64,
    min_samples: i32,
    min_cluster_size: i32,
    metric: &str,
    boruvka: bool,
    partial_labels: Nullable<List>,
) -> List {
    let data = rmatrix_to_array2(x);
    let n_points = data.nrows();
    let min_samples = min_samples as usize;
    let min_cluster_size = min_cluster_size as usize;

    let labels = match partial_labels {
        Nullable::NotNull(pl) => Some(partial_labels_from_list(pl)),
        Nullable::Null => None,
    };

    let (clusters, scores) = threads::pool().install(|| match metric {
        "euclidean" => {
            let mut model = HDbscan {
                alpha,
                min_samples,
                min_cluster_size,
                metric: Euclidean::default(),
                boruvka,
            };
            let (clusters, _noise, scores) = model.fit(&data, labels.as_ref());
            (clusters, scores)
        }
        "cosine" => {
            let mut model = HDbscan {
                alpha,
                min_samples,
                min_cluster_size,
                metric: Cosine::default(),
                boruvka,
            };
            let (clusters, _noise, scores) = model.fit(&data, labels.as_ref());
            (clusters, scores)
        }
        _ => panic!("Unknown metric: {metric}"),
    });

    let assignment = assignment_vector(&clusters, n_points);
    let outlier_scores: Vec<Rfloat> = scores.iter().map(|&s| Rfloat::from(s)).collect();

    list!(cluster = assignment, outlier_scores = outlier_scores)
}

/// Row-major copy of an R matrix: rows are then contiguous, where R's
/// column-major layout would stride through memory once per feature.
fn row_major(x: &RMatrix<f64>) -> Vec<f64> {
    let n = x.nrows();
    let ncol = x.ncols();
    let col_major = x.data();
    let mut data = Vec::with_capacity(n * ncol);
    for i in 0..n {
        for c in 0..ncol {
            data.push(col_major[c * n + i]);
        }
    }
    data
}

// Condensed lower-triangle distance matrix, in R's `dist` layout.
//
// The result vector is allocated by R up front and filled in place, so the
// n(n-1)/2 distances are written exactly once; at 20,000 points that run is
// 1.6 GB, and every intermediate copy of it used to cost more than the
// arithmetic on narrow data.
//
// Returns the values and whether they are all finite, so the R side need not
// scan the result again.
#[extendr]
fn rust_dist(x: RMatrix<f64>, metric: &str, p: f64) -> List {
    let n = x.nrows();
    let ncol = x.ncols();
    let metric = dist::Metric::from_name(metric, p);
    let data = row_major(&x);

    let mut out = Doubles::new(n * (n.saturating_sub(1)) / 2);
    let finite = {
        let slice: &mut [Rfloat] = &mut out;
        threads::pool().install(|| dist::condensed_into(&data, n, ncol, metric, slice))
    };
    list!(values = out, finite = finite)
}

// Exact k nearest neighbours of each row of `x` (self excluded), or of each
// row of `query` among the rows of `x`, by a brute-force scan or a kd-tree.
// Both matrices come across column-major and are made row-major once, so
// the inner loop over candidates walks contiguous memory.
//
// Returns one-based indices and distances as `m x k` matrices, plus whether
// every distance computed was finite, so the R side need not scan them.
#[extendr]
fn rust_knn(
    x: RMatrix<f64>,
    query: Nullable<RMatrix<f64>>,
    k: i32,
    metric: &str,
    p: f64,
    search: &str,
) -> List {
    let n = x.nrows();
    let ncol = x.ncols();
    let k = k as usize;
    let metric = dist::Metric::from_name(metric, p);
    let search = knn::Search::from_name(search);
    let data = row_major(&x);
    let query = match query {
        Nullable::NotNull(q) => Some(row_major(&q)),
        Nullable::Null => None,
    };

    let res = threads::pool()
        .install(|| knn::search(&data, n, ncol, query.as_deref(), k, metric, search));

    let m = res.index.len() / k;
    let id = RMatrix::new_matrix(m, k, |i, j| res.index[i * k + j]);
    let dist = RMatrix::new_matrix(m, k, |i, j| res.dist[i * k + j]);
    list!(id = id, dist = dist, finite = res.finite)
}

// Hierarchical clustering straight from data: Euclidean distances are
// written into the one buffer kodama consumes, so the n(n-1)/2 values exist
// once rather than as an R `dist` plus the copy kodama needs. At 50,000
// points that is 10 GB instead of 20.
#[extendr]
fn rust_hclust_data(x: RMatrix<f64>, method: &str) -> List {
    let n = x.nrows();
    let ncol = x.ncols();
    let data = row_major(&x);

    let mut d = vec![0.0f64; n * (n.saturating_sub(1)) / 2];
    let finite = threads::pool()
        .install(|| dist::condensed_into(&data, n, ncol, dist::Metric::Euclidean, &mut d));
    if !finite {
        throw_r_error("Distance computation produced non-finite values.");
    }

    hclust_list(d, n, method)
}

// Hierarchical clustering over a condensed dissimilarity matrix.
//
// Returns the `merge`, `height` and `order` components of an `hclust` object.
// `d` is the R vector itself (a `dist`, attributes and all); it is copied
// exactly once here, because kodama destroys the matrix it is given.
#[extendr]
fn rust_hclust(d: Robj, n: i32, method: &str) -> List {
    let n = n as usize;
    let d: Vec<f64> = d
        .as_real_slice()
        .expect("`d` must be a double vector")
        .to_vec();

    let expected = n * (n - 1) / 2;
    if d.len() != expected {
        panic!("expected {expected} dissimilarities for {n} observations, got {}", d.len());
    }
    // kodama panics on NaN and misbehaves on Inf. The R side screens NA
    // cheaply; the full finiteness scan lives here, where it is one pass over
    // memory that is about to be read anyway.
    if d.iter().any(|v| !v.is_finite()) {
        throw_r_error("`d` must not contain missing or non-finite values.");
    }

    hclust_list(d, n, method)
}

/// Run kodama on a condensed matrix it may destroy and package the result as
/// the `merge`, `height` and `order` components of an `hclust` object.
fn hclust_list(d: Vec<f64>, n: usize, method: &str) -> List {
    let out = hclust::hclust(d, n, hclust::method_from_name(method));
    let n_steps = out.height.len();

    let merge = RMatrix::new_matrix(n_steps, 2, |r, c| out.merge[c * n_steps + r]);

    list!(merge = merge, height = out.height, order = out.order)
}

// k-means clustering.
//
// Returns the assignment vector, centroids, inertia and cluster sizes.
#[extendr]
fn rust_kmeans(
    x: RMatrix<f64>,
    k: i32,
    init: &str,
    n_runs: i32,
    max_iter: i32,
    tolerance: f64,
    seed: f64,
) -> List {
    let data = rmatrix_to_array2_linfa(x);

    // linfa's assignment step and initialisation are parallel through
    // ndarray's rayon feature, so they run in the package pool like
    // everything else. extendr 0.8 turns an `Err` return into an opaque
    // "User function panicked", so errors are raised explicitly to keep
    // their message.
    let init = kmeans::init_from_name(init);
    let fit = unwrap_or_throw_error(
        threads::pool()
            .install(|| {
                kmeans::fit(
                    data,
                    k as usize,
                    init,
                    n_runs as usize,
                    max_iter as u64,
                    tolerance,
                    seed as u64,
                )
            })
            .map_err(|e| Error::Other(format!("k-means failed: {e}"))),
    );

    let n_clusters = fit.centroids.nrows();
    let n_features = fit.centroids.ncols();

    let cluster: Vec<Rint> = fit
        .assignments
        .iter()
        .map(|&c| Rint::from((c + 1) as i32)) // 1-indexed for R
        .collect();

    let centroids = RMatrix::new_matrix(n_clusters, n_features, |r, c| fit.centroids[[r, c]]);

    list!(
        cluster = cluster,
        centroids = centroids,
        inertia = fit.inertia,
        sizes = fit.sizes
    )
}

// Assign observations to their nearest centroid. Backs `predict()`.
#[extendr]
fn rust_nearest_centroid(x: RMatrix<f64>, centroids: RMatrix<f64>) -> Integers {
    let data = rmatrix_to_array2_linfa(x);
    let centroids = rmatrix_to_array2_linfa(centroids);

    threads::pool()
        .install(|| kmeans::nearest_centroid(&data, &centroids))
        .into_iter()
        .map(Rint::from)
        .collect()
}

// Gaussian mixture model.
//
// Returns only the fitted parameters; responsibilities, assignments and the
// log-likelihood are derived from them on the R side.
#[extendr]
fn rust_gmm(
    x: RMatrix<f64>,
    k: i32,
    init: &str,
    n_runs: i32,
    max_iter: i32,
    tolerance: f64,
    reg_covariance: f64,
    seed: f64,
) -> List {
    let data = rmatrix_to_array2_linfa(x);

    let init = gmm::init_from_name(init);
    let fit = unwrap_or_throw_error(
        threads::pool()
            .install(|| {
                gmm::fit(
                    data,
                    k as usize,
                    init,
                    n_runs as u64,
                    max_iter as u64,
                    tolerance,
                    reg_covariance,
                    seed as u64,
                )
            })
            .map_err(|e| Error::Other(format!("Gaussian mixture fit failed: {e}"))),
    );

    let n_clusters = fit.means.nrows();
    let n_features = fit.means.ncols();
    let means = RMatrix::new_matrix(n_clusters, n_features, |r, c| fit.means[[r, c]]);

    list!(
        weights = fit.weights,
        means = means,
        covariances = fit.covariances
    )
}

// EVoC: direct multi-layer clustering of embedding vectors.
//
// Returns every cluster layer (finest first) with membership strengths and
// persistence scores, plus the learned node embedding; which layer to surface
// as `cluster` is the R side's decision. `dim = 0` means the reference
// default `min(max(n_neighbors / 4, 4), 15)`.
#[extendr]
fn rust_evoc(
    x: RMatrix<f64>,
    n_neighbors: i32,
    noise_level: f64,
    min_cluster_size: i32,
    min_samples: i32,
    n_epochs: i32,
    dim: i32,
    min_similarity_threshold: f64,
    max_layers: i32,
    n_label_prop_iter: i32,
    seed: f64,
) -> List {
    let n = x.nrows();
    let (data, _, p) = unwrap_or_throw_error(evoc::rmatrix_to_rowmajor_f32(&x));
    let result = threads::pool().install(|| {
        evoc::run(
            &data,
            p,
            n_neighbors as usize,
            noise_level,
            i64::from(min_cluster_size),
            min_samples as usize,
            n_epochs as usize,
            if dim > 0 { Some(dim as usize) } else { None },
            min_similarity_threshold,
            max_layers as usize,
            n_label_prop_iter as usize,
            seed as u64,
        )
    });
    evoc::result_to_list(&result, n)
}

// Per-observation silhouette widths from a condensed distance matrix.
//
// `d` is read in place from the R vector (a `dist`): at 20,000 observations
// it is 1.6 GB, and it is only ever read. `cluster` arrives 1-indexed from R
// and is returned to R the same way.
#[extendr]
fn rust_silhouette(d: Robj, n: i32, cluster: Vec<i32>, k: i32) -> List {
    let n = n as usize;
    let k = k as usize;
    let d: &[f64] = d.as_real_slice().expect("`d` must be a double vector");
    let zero_based: Vec<usize> = cluster.iter().map(|&c| (c - 1) as usize).collect();

    let (widths, neighbours) =
        threads::pool().install(|| metrics::silhouette(d, n, &zero_based, k));

    // usize::MAX marks "no neighbouring cluster"; that becomes NA in R.
    let neighbour: Vec<Rint> = neighbours
        .into_iter()
        .map(|nb| {
            if nb == usize::MAX {
                Rint::na()
            } else {
                Rint::from((nb + 1) as i32)
            }
        })
        .collect();

    list!(width = widths, neighbour = neighbour)
}

// Calinski-Harabasz and Davies-Bouldin indices.
#[extendr]
fn rust_cluster_indices(x: RMatrix<f64>, cluster: Vec<i32>, k: i32) -> List {
    let data = rmatrix_to_array2(x);
    let n = data.nrows();
    let p = data.ncols();
    let k = k as usize;

    // Row-major copy: the metrics walk observations, not features.
    let mut flat = Vec::with_capacity(n * p);
    for i in 0..n {
        for f in 0..p {
            flat.push(data[[i, f]]);
        }
    }

    let zero_based: Vec<usize> = cluster.iter().map(|&c| (c - 1) as usize).collect();

    list!(
        calinski_harabasz = metrics::calinski_harabasz(&flat, n, p, &zero_based, k),
        davies_bouldin = metrics::davies_bouldin(&flat, n, p, &zero_based, k)
    )
}

// Rebuild the package thread pool with `n` threads.
#[extendr]
fn rust_set_threads(n: i32) {
    threads::set_threads(n.max(1) as usize);
}

// Threads in the package pool.
#[extendr]
fn rust_get_threads() -> i32 {
    threads::threads() as i32
}


// Graph SLIC superpixels. `nb` is an integer matrix of one-based neighbour
// indices with NA for a missing slot; `init` one-based dense labels.
#[extendr]
fn rust_slic(
    x: RMatrix<f64>,
    xy: RMatrix<f64>,
    nb: RMatrix<i32>,
    init: Vec<i32>,
    lambda: f64,
    s_nom: f64,
    n_iter: i32,
    adaptive: &str,
    alpha: f64,
    min_patch: i32,
    tol: f64,
) -> List {
    let n = x.nrows();
    let d = x.ncols();
    let k = nb.ncols();
    let xr = row_major(&x);
    let xyr = row_major(&xy);
    let nb_col = nb.data();
    let mut nbr = vec![-1i32; n * k];
    for i in 0..n {
        for c in 0..k {
            let v = nb_col[c * n + i];
            nbr[i * k + c] = if v == i32::MIN || v < 1 { -1 } else { v - 1 };
        }
    }
    let init0: Vec<i32> = init.iter().map(|&l| l - 1).collect();
    let params = slic::Params {
        lambda,
        s_nom,
        n_iter: n_iter as usize,
        adaptive: slic::Adaptive::from_name(adaptive),
        alpha,
        min_patch: min_patch as usize,
        tol,
    };
    let g = slic::Graph { nb: &nbr, k };
    let out = threads::pool().install(|| slic::graph_slic(&xr, d, &xyr, &g, &init0, &params));
    list!(
        labels = Integers::from_values(out.labels.iter().map(|&l| Rint::from(l))),
        moves = Integers::from_values(out.moves.iter().map(|&m| Rint::from(m))),
        init_stat = Doubles::from_values(out.init_stat.iter().copied()),
        lambda_median = Doubles::from_values(out.lambda_median.iter().copied()),
        n_patches = out.n_patches as i32
    )
}


// Two-sample energy distance: mean cross-pair and within-sample distances.
#[extendr]
fn rust_energy(x: RMatrix<f64>, y: RMatrix<f64>) -> List {
    let d = x.ncols();
    let xr = row_major(&x);
    let yr = row_major(&y);
    let e = threads::pool().install(|| energy::energy(&xr, &yr, d));
    list!(cross = e.cross, self_x = e.self_x, self_y = e.self_y)
}

extendr_module! {
    mod shoal;
    fn rust_set_threads;
    fn rust_get_threads;
    fn rust_dbscan;
    fn rust_hdbscan;
    fn rust_dist;
    fn rust_knn;
    fn rust_hclust;
    fn rust_hclust_data;
    fn rust_kmeans;
    fn rust_nearest_centroid;
    fn rust_gmm;
    fn rust_evoc;
    fn rust_silhouette;
    fn rust_cluster_indices;
    fn rust_slic;
    fn rust_energy;
}
