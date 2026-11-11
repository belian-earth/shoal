use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::{KMeans, KMeansInit};
use linfa_nn::distance::{Distance, L2Dist};
use ndarray_linfa::{Array1, Array2, ArrayView2, Axis, Zip};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;

/// Resolve an initialisation strategy by name.
///
/// Names are validated on the R side; an unknown name here is a bug.
pub fn init_from_name(name: &str) -> KMeansInit<f64> {
    match name {
        "kmeans++" => KMeansInit::KMeansPlusPlus,
        "kmeans_parallel" => KMeansInit::KMeansPara,
        "random" => KMeansInit::Random,
        _ => panic!("Unknown initialisation method: {name}"),
    }
}

/// A fitted k-means model, reduced to the pieces R needs.
pub struct KMeansFit {
    /// Zero-based cluster index per observation.
    pub assignments: Array1<usize>,
    /// `k x n_features` centroid matrix.
    pub centroids: Array2<f64>,
    /// Sum of squared distances of each point to its assigned centroid.
    pub inertia: f64,
    /// Number of observations assigned to each cluster.
    pub sizes: Vec<i32>,
}

/// A start drawn by shoal: `k` centres from the rows, consuming `rng`.
type Start = fn(ArrayView2<f64>, usize, &mut Xoshiro256Plus) -> Array2<f64>;

/// Fit k-means.
///
/// `seed` makes the run reproducible: linfa takes an explicit RNG, so the same
/// seed and parameters always give the same partition.
///
/// The k-means++ and k-means|| starts are drawn here rather than by linfa.
/// linfa's k-means++ start costs O(n k^2 d) (see [`kmeans_plusplus`]), and
/// its k-means|| start depends on the thread count (see [`kmeans_parallel`]).
/// The restarts then run here too, each handing its start to linfa for the
/// Lloyd iterations; the random start is still linfa's. For k-means++ the
/// random draws, the iterations and the choice of run are linfa's, so the
/// result is the one linfa's own k-means++ would give.
pub fn fit(
    data: Array2<f64>,
    k: usize,
    init: KMeansInit<f64>,
    n_runs: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> Result<KMeansFit, String> {
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    let dataset = DatasetBase::from(data);

    let lloyd = |init: KMeansInit<f64>, n_runs: usize, rng: Xoshiro256Plus| {
        KMeans::params_with_rng(k, rng)
            .n_runs(n_runs)
            .max_n_iterations(max_iter)
            .tolerance(tolerance)
            .init_method(init)
            .fit(&dataset)
            .map_err(|e| e.to_string())
    };

    let start: Option<Start> = match init {
        KMeansInit::KMeansPlusPlus => Some(|x, k, rng| kmeans_plusplus(x, None, k, rng)),
        KMeansInit::KMeansPara => Some(kmeans_parallel),
        _ => None,
    };

    let model = match start {
        Some(start) => {
            // linfa keeps the first run of lowest inertia. It compares the
            // sums and reports the mean; dividing every run's sum by the
            // same n preserves the order, short of two sums within rounding
            // of each other.
            let mut best: Option<KMeans<f64, L2Dist>> = None;
            for _ in 0..n_runs {
                let start = start(dataset.records().view(), k, &mut rng);
                let model = lloyd(KMeansInit::Precomputed(start), 1, rng.clone())?;
                if best
                    .as_ref()
                    .map_or(true, |b| model.inertia() < b.inertia())
                {
                    best = Some(model);
                }
            }
            best.ok_or_else(|| "no k-means run completed".to_string())?
        }
        None => lloyd(init, n_runs, rng)?,
    };

    let assignments: Array1<usize> = model.predict(dataset.records());
    let centroids = model.centroids().to_owned();

    // Not taken from the model: linfa's `inertia()` is the *mean* squared
    // distance, measured one iteration before the final centroid update, and
    // `cluster_count()` comes from whichever run happened last rather than
    // the best one. Both are derived here from the assignments actually
    // returned, so the three components always describe the same partition.
    let mut sizes = vec![0i32; k];
    let mut inertia = 0.0f64;
    for (row, &c) in dataset.records().rows().into_iter().zip(assignments.iter()) {
        sizes[c] += 1;
        inertia += row
            .iter()
            .zip(centroids.row(c).iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>();
    }

    Ok(KMeansFit {
        assignments,
        centroids,
        inertia,
        sizes,
    })
}

/// Margin on the triangle-inequality skip in [`kmeans_plusplus`]. Computed
/// squared distances carry a relative error of about `d` units in the last
/// place, near 1e-14 at 64 columns; a margin a million times larger keeps a
/// skipped row's computed distance strictly above its minimum.
const SKIP_MARGIN: f64 = 1e-8;

/// The k-means++ start (Arthur and Vassilvitskii 2007), drawn as
/// linfa-clustering 0.8.1 draws it.
///
/// Each centre after the first is a row drawn with probability proportional
/// to its squared distance from the nearest centre chosen so far, times its
/// weight when `weights` are given, as for the candidates of k-means||. linfa
/// recomputes that distance against every chosen centre at each step, so its
/// start costs O(n k^2 d), as much as k/2 Lloyd iterations. Here each row
/// keeps a running minimum and meets only the newest centre, for O(n k d) in
/// all. A minimum takes no arithmetic, so the distances, and with them the
/// draws from `rng`, are bit for bit linfa's: the same `rand` sampler is fed
/// the same weights in the same order.
///
/// Each step would still read every row to meet one centre. The triangle
/// inequality skips a row when the newest centre is more than twice as far
/// from the row's nearest centre as that centre is from the row: it cannot
/// be nearer, and the row's minimum stands. On AlphaEarth embeddings this
/// halves the start; where most rows are about equally far from every
/// centre, as when k is below the number of natural groups, it skips few.
fn kmeans_plusplus(
    observations: ArrayView2<f64>,
    weights: Option<&[f64]>,
    k: usize,
    rng: &mut Xoshiro256Plus,
) -> Array2<f64> {
    let (n, p) = observations.dim();
    let mut centroids = Array2::zeros((k, p));

    // linfa draws the first centre by weight, through a sampler over unit
    // weights when there are none.
    let first = match weights {
        Some(w) => WeightedIndex::new(w),
        None => WeightedIndex::new(std::iter::repeat(1.0f64).take(n)),
    }
    .expect("at least one row of positive weight")
    .sample(rng);
    centroids.row_mut(0).assign(&observations.row(first));

    let mut dists = Array1::from_elem(n, f64::INFINITY);
    // The chosen centre at each row's minimum, and every chosen centre's
    // squared distance from the newest.
    let mut nearest = Array1::<u32>::zeros(n);
    let mut from_newest = vec![f64::INFINITY; k];
    for c in 1..k {
        let newest = centroids.row(c - 1);
        for (j, slot) in from_newest.iter_mut().enumerate().take(c - 1) {
            *slot = L2Dist.rdistance(centroids.row(j), newest);
        }
        from_newest[c - 1] = 0.0;
        let from_newest = &from_newest;
        Zip::from(&mut dists)
            .and(&mut nearest)
            .and(observations.rows())
            .par_for_each(|dist, near, row| {
                // A row yet to meet a finite distance is never skipped.
                if *dist < f64::INFINITY
                    && from_newest[*near as usize] >= 4.0 * *dist * (1.0 + SKIP_MARGIN)
                {
                    return;
                }
                let d = L2Dist.rdistance(newest, row);
                if d < *dist {
                    *dist = d;
                    *near = (c - 1) as u32;
                }
            });
        // Every row on a chosen centre leaves no weight to draw by. Any row
        // then serves, and linfa takes the first. Unit weights would change
        // nothing, so the unweighted draw skips the product.
        let sampler = match weights {
            Some(w) => WeightedIndex::new(dists.iter().zip(w).map(|(d, w)| d * w)),
            None => WeightedIndex::new(dists.iter()),
        };
        let next = sampler.map(|w| w.sample(rng)).unwrap_or(0);
        centroids.row_mut(c).assign(&observations.row(next));
    }
    centroids
}

/// Rows per random stream in [`kmeans_parallel`]. Fixed, so the draws do
/// not depend on how rows are shared among threads.
const SAMPLE_BLOCK: usize = 4096;

/// The k-means|| start (Bahmani et al. 2012), with linfa-clustering 0.8.1's
/// settings: eight rounds, each drawing about k candidates, every row with
/// probability k times its share of the total squared distance to the
/// candidates so far, until 8k are drawn. The candidates, weighted by the
/// rows nearest each, are then reduced to k centres by weighted k-means++.
///
/// linfa draws each round from one random stream per piece of work handed to
/// a thread, and the pieces vary with the thread count and with timing, so
/// its start is not reproducible from a seed. Here each block of
/// [`SAMPLE_BLOCK`] rows has its own stream, seeded from the round's seed
/// and the block's position. linfa also measures every row against every
/// candidate at each round; here each round meets only the candidates new
/// since the last, and the pass that meets the final round's candidates
/// also finds each row's nearest candidate for the weights.
fn kmeans_parallel(
    observations: ArrayView2<f64>,
    k: usize,
    rng: &mut Xoshiro256Plus,
) -> Array2<f64> {
    const ROUNDS: usize = 8;
    let n = observations.nrows();
    let capacity = k * ROUNDS;
    let per_round = k as f64;

    let mut candidates = vec![rng.gen_range(0..n)];
    let mut dists = Array1::from_elem(n, f64::INFINITY);
    let mut nearest = Array1::<u32>::zeros(n);
    let mut met = 0;
    'rounds: for _ in 0..ROUNDS {
        meet(observations, &candidates, met, &mut dists, &mut nearest);
        met = candidates.len();

        let cost: f64 = dists.iter().sum();
        if cost.is_nan() || cost <= 0.0 {
            // Every row is a candidate already.
            break;
        }
        let round_seed: u64 = rng.gen();
        let dists = dists.as_slice().expect("contiguous");
        let drawn: Vec<usize> = dists
            .par_chunks(SAMPLE_BLOCK)
            .enumerate()
            .flat_map_iter(|(block, chunk)| {
                let mut stream =
                    Xoshiro256Plus::seed_from_u64(round_seed.wrapping_add(block as u64));
                let offset = block * SAMPLE_BLOCK;
                chunk
                    .iter()
                    .enumerate()
                    .filter(move |(_, &d)| stream.gen::<f64>() < per_round * d / cost)
                    .map(move |(i, _)| offset + i)
            })
            .collect();
        for row in drawn {
            candidates.push(row);
            if candidates.len() >= capacity {
                break 'rounds;
            }
        }
    }
    meet(observations, &candidates, met, &mut dists, &mut nearest);

    let mut weights = vec![0.0f64; candidates.len()];
    for &c in nearest.iter() {
        weights[c as usize] += 1.0;
    }
    let pool = observations.select(Axis(0), &candidates);
    kmeans_plusplus(pool.view(), Some(&weights), k, rng)
}

/// Lower each row's squared distance to its nearest candidate by meeting
/// the candidates from position `from` on, recording the nearest. Ties go
/// to the earlier candidate.
fn meet(
    observations: ArrayView2<f64>,
    candidates: &[usize],
    from: usize,
    dists: &mut Array1<f64>,
    nearest: &mut Array1<u32>,
) {
    let fresh = observations.select(Axis(0), &candidates[from..]);
    Zip::from(dists)
        .and(nearest)
        .and(observations.rows())
        .par_for_each(|dist, near, row| {
            for (j, candidate) in fresh.rows().into_iter().enumerate() {
                let d = L2Dist.rdistance(candidate, row);
                if d < *dist {
                    *dist = d;
                    *near = (from + j) as u32;
                }
            }
        });
}

/// Assign observations to their nearest centroid, by squared Euclidean distance.
///
/// Used by `predict()` on a fitted model. Kept independent of linfa so a saved
/// model is just its centroids -- there is no Rust-side state to keep alive
/// between calls.
pub fn nearest_centroid(data: &Array2<f64>, centroids: &Array2<f64>) -> Vec<i32> {
    data.rows()
        .into_iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (idx, centroid) in centroids.rows().into_iter().enumerate() {
                let dist: f64 = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = idx;
                }
            }
            (best + 1) as i32 // 1-indexed for R
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// k-means as it ran before: linfa's own start and restarts.
    fn fit_linfa(data: &Array2<f64>, k: usize, n_runs: usize, seed: u64) -> KMeans<f64, L2Dist> {
        KMeans::params_with_rng(k, Xoshiro256Plus::seed_from_u64(seed))
            .n_runs(n_runs)
            .max_n_iterations(300)
            .tolerance(1e-4)
            .init_method(KMeansInit::KMeansPlusPlus)
            .fit(&DatasetBase::from(data.clone()))
            .unwrap()
    }

    fn blobs(n: usize, p: usize, seed: u64) -> Array2<f64> {
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let centres: Vec<f64> = (0..8 * p).map(|_| rng.gen_range(-10.0..10.0)).collect();
        Array2::from_shape_fn((n, p), |(i, j)| {
            centres[(i % 8) * p + j] + rng.gen_range(-1.0..1.0)
        })
    }

    #[test]
    fn kmeans_plusplus_reproduces_linfa_bit_for_bit() {
        for &(n, p, k, n_runs, seed) in &[
            (400, 3, 5, 1, 1),
            (400, 3, 5, 10, 2),
            (1000, 7, 40, 3, 3),
            (257, 2, 64, 4, 4),
            (500, 16, 1, 2, 5),
        ] {
            let x = blobs(n, p, seed);
            let want = fit_linfa(&x, k, n_runs, seed);
            let got = fit(x, k, KMeansInit::KMeansPlusPlus, n_runs, 300, 1e-4, seed).unwrap();
            assert_eq!(
                &got.centroids,
                want.centroids(),
                "n {n}, p {p}, k {k}, runs {n_runs}"
            );
        }
    }

    #[test]
    fn skipped_rows_leave_the_draws_unchanged() {
        // Many centres, so most rows are skipped at most steps; data far
        // from the origin, where rounding is largest against the spread; and
        // an integer lattice with repeated rows, where distances tie exactly.
        let offset = blobs(2000, 6, 11).mapv(|v| v + 1e6);
        let lattice = Array2::from_shape_fn((1500, 3), |(i, j)| ((i * (j + 7)) % 5) as f64);
        for (x, k, seed) in [
            (blobs(2000, 6, 10), 300, 10),
            (offset, 200, 11),
            (lattice, 90, 12),
        ] {
            let want = fit_linfa(&x, k, 2, seed);
            let got = fit(x, k, KMeansInit::KMeansPlusPlus, 2, 300, 1e-4, seed).unwrap();
            assert_eq!(&got.centroids, want.centroids(), "k {k}, seed {seed}");
        }
    }

    #[test]
    fn kmeans_plusplus_matches_linfa_when_centres_run_out() {
        // Four distinct rows for six centres: the last draws have no weight.
        let x = Array2::from_shape_fn((40, 2), |(i, j)| ((i % 4) * 10 + j) as f64);
        let want = fit_linfa(&x, 6, 2, 9);
        let got = fit(x, 6, KMeansInit::KMeansPlusPlus, 2, 300, 1e-4, 9).unwrap();
        assert_eq!(&got.centroids, want.centroids());
    }

    #[test]
    fn kmeans_plusplus_does_not_depend_on_the_thread_count() {
        let x = blobs(3000, 5, 7);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    kmeans_plusplus(x.view(), None, 50, &mut Xoshiro256Plus::seed_from_u64(7))
                })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn kmeans_parallel_does_not_depend_on_the_thread_count() {
        // Several sampling blocks, so threads share the draws.
        let x = blobs(20_000, 4, 8);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| kmeans_parallel(x.view(), 40, &mut Xoshiro256Plus::seed_from_u64(8)))
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert_eq!(one, run(7));
        assert_ne!(
            one,
            kmeans_parallel(x.view(), 40, &mut Xoshiro256Plus::seed_from_u64(9))
        );
    }

    #[test]
    fn kmeans_parallel_finds_separated_groups() {
        // Eight tight, distant groups: a sound start puts one centre in each,
        // and Lloyd then lands on the group means, as from k-means++.
        let noise = blobs(4000, 3, 13);
        let x = Array2::from_shape_fn((4000, 3), |(i, j)| {
            ((i % 8) * 100 + j) as f64 + 0.01 * noise[[i, j]]
        });
        let para = fit(x.clone(), 8, KMeansInit::KMeansPara, 1, 300, 1e-4, 13).unwrap();
        let plus = fit(x, 8, KMeansInit::KMeansPlusPlus, 1, 300, 1e-4, 13).unwrap();
        let mut sizes = para.sizes.clone();
        sizes.sort();
        assert_eq!(sizes, vec![500; 8]);
        assert!((para.inertia - plus.inertia).abs() <= 1e-9 * plus.inertia);
    }

    #[test]
    fn kmeans_parallel_handles_fewer_distinct_rows_than_centres() {
        let x = Array2::from_shape_fn((40, 2), |(i, j)| ((i % 4) * 10 + j) as f64);
        let start = kmeans_parallel(x.view(), 6, &mut Xoshiro256Plus::seed_from_u64(3));
        assert_eq!(start.dim(), (6, 2));
        let fit = fit(x, 6, KMeansInit::KMeansPara, 3, 300, 1e-4, 3).unwrap();
        assert_eq!(fit.sizes.iter().sum::<i32>(), 40);
    }
}
