//! e-means: mini-batch k-means by exponential moving averages.
//!
//! Ported from the reference implementation in faiss, `contrib/e_means`
//! (<https://github.com/facebookresearch/faiss/tree/main/contrib/e_means>),
//! by Gergely Szilvasy, Swann Bessa and Matthijs Douze at Meta FAIR, at
//! upstream commit f777f7d. The Euclidean path follows `NumpyEmeans` and the
//! cosine path its `spherical=True` mode. The method is Algorithm 1 of
//! Bessa et al. (2026), "e-means: outperforming k-means (in one-tenth the
//! epochs)", arXiv:2611.09501.
//!
//! The reference is distributed under the MIT licence:
//!
//! > Copyright (c) Meta Platforms, Inc. and affiliates.
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a
//! > copy of this software and associated documentation files (the
//! > "Software"), to deal in the Software without restriction, including
//! > without limitation the rights to use, copy, modify, merge, publish,
//! > distribute, sublicense, and/or sell copies of the Software, and to
//! > permit persons to whom the Software is furnished to do so, subject to
//! > the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included
//! > in all copies or substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//! > OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//! > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//! > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//! > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//! > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//! > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! # The algorithm
//!
//! Each epoch visits the rows in a fresh random order, in batches of `B`.
//! A batch is assigned to the current centroids, giving per-centroid counts
//! `n_j` and sums `s_j`, and every centroid with points moves towards its
//! batch mean by an exponential moving average over a running mass `N_j`:
//!
//! ```text
//! N_j <- (1 - a) N_j + a n_j
//! c_j <- c_j + (a n_j / N_j) (s_j / n_j - c_j)
//! ```
//!
//! The step `a` starts at `alpha0` and is cosine-annealed to a tenth of it
//! over the run. `B` and `alpha0` come from scaling laws the authors fitted
//! on n, k and the epoch budget; see [`recipe`]. Every five updates,
//! centroids whose mass has fallen below a fraction of the mean are revived
//! by splitting a donor drawn in proportion to its mass. The returned
//! centroids are the plain means of the points assigned to them during the
//! final epoch.
//!
//! Exact nearest-centroid search is the only heavy step. It is the argmin
//! over `j` of `|c_j|^2 - 2 x.c_j`, whose dot products are a matrix product,
//! formed in tiles by gemm (matrixmultiply, through ndarray) and reduced to
//! a running minimum as each tile is formed. Cosine mode takes the maximum
//! inner product against the unit-length centroid directions instead.
//!
//! # Parity and divergences
//!
//! `emeans_fixtures.rs`, generated from the reference by
//! `data-raw/emeans_fixtures.py`, fixes the recipe for a grid of problem
//! sizes, and the trained centroids for small runs with a given start, batch
//! order and stream of revival draws. Divergences from the reference, all
//! deliberate:
//!
//! - Double precision throughout, where the reference keeps its model in
//!   single precision.
//! - The random streams are shoal's own (xoshiro256++), so a seed does not
//!   reproduce the reference's draws; the fixtures inject them instead.
//! - Cosine mode normalises the input rows. The reference expects the
//!   caller to.
//! - Euclidean mode trains on the rows less their column means, and makes
//!   its final assignment relative to the mean of the centroids. Neither
//!   changes the k-means objective. Without them, both the revival
//!   perturbation, which is relative to the coordinates, and the rounding
//!   in `|c|^2 - 2 x.c` depend on where the data sits: far from the origin,
//!   as projected coordinates are, splits stop separating and rows land on
//!   the wrong centroid. The reference works in raw coordinates.
//! - The search ranks centroids by `|c|^2 - 2 x.c` without the row's own
//!   `|x|^2`, where the reference ranks by the full squared distance. The
//!   two orders agree except where the full distance rounds two centroids
//!   to a tie. The visible case is a donor exactly at the (centred) origin:
//!   its split pair stays separable here, while the reference ties them,
//!   gives every row to the first, and revives the second again.
//! - [`fit`] ends with one exact assignment pass, so the labels are each
//!   row's nearest final centroid. The reference returns the labels from
//!   the final training pass, while the centroids were still moving.
//!
//! Results do not depend on the thread count: assignment tiles are fixed
//! by row position, and every sum is formed in a fixed order. They can
//! differ between machines, because matrixmultiply picks its kernel (with
//! or without fused multiply-add) for the CPU it runs on.

use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2};
use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Epsilon in Equation 5: the annealed step's floor, as a fraction of `alpha0`.
const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Updates between revival checks (Algorithm 1, line 9).
const CHECK_UNUSED_EVERY: usize = 5;
/// Initial mass of every centroid. The reference uses 2^-16; the paper 1e-10.
const INIT_USAGE: f64 = 1.0 / 65536.0;
/// Relative coordinate perturbation when a donor is split (Appendix J).
const SPLIT_EPSILON: f64 = 1e-4;
/// Default revival threshold, as a fraction of the mean mass.
const UNUSED_THRESHOLD: f64 = 0.1;
/// At or below this dimension the default turns revival off.
const NO_REVIVAL_MAX_D: usize = 2;
/// At or below this dimension the automatic batch law is the quality law.
const AUTO_QUALITY_MAX_D: usize = 16;
/// Batch law `B = C k^bk n^bn T^bT` as `(C, bk, bn, bT)` (Equation 7). The
/// quality law is the throughput law with an eighth of its coefficient.
const THROUGHPUT_LAW: [f64; 4] = [0.035134, 0.134749, 0.876473, 0.813344];
const QUALITY_LAW: [f64; 4] = [0.00439175, 0.134749, 0.876473, 0.813344];
/// Step law `alpha0 = C k^ak B^aB n^an T^aT` as `(C, ak, aB, an, aT)`
/// (Equation 6). As the reference notes, the shipped coefficients of both
/// laws differ from the paper's Table 2.
const ALPHA_LAW: [f64; 5] = [0.8244, -0.4617, 0.9003, -0.4498, -0.1057];

/// Rows per assignment tile. Small, so that tiles outnumber threads.
const ROW_BLOCK: usize = 64;
/// Centroids per assignment tile; with [`ROW_BLOCK`], a 512 KB product.
const COL_BLOCK: usize = 1024;
/// Rows per block when column means are summed, in a fixed order.
const MEAN_BLOCK: usize = 4096;
/// Upper bound on the tasks that accumulate a batch, each owning a range of
/// centroids.
const ACCUMULATE_TASKS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchLaw {
    Auto,
    Throughput,
    Quality,
}

impl BatchLaw {
    /// Names are validated on the R side; an unknown name here is a bug.
    pub fn from_name(name: &str) -> Self {
        match name {
            "auto" => BatchLaw::Auto,
            "throughput" => BatchLaw::Throughput,
            "quality" => BatchLaw::Quality,
            _ => panic!("Unknown batch law: {name}"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BatchLaw::Auto => "auto",
            BatchLaw::Throughput => "throughput",
            BatchLaw::Quality => "quality",
        }
    }
}

/// The resolved training plan for one run.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// The batch law that set `batch_size`, with `Auto` resolved; `None`
    /// when the batch size was given, as in the reference.
    pub law: Option<BatchLaw>,
    pub batch_size: usize,
    pub alpha0: f64,
    /// Rows visited in each epoch: whole passes, then any fractional prefix,
    /// then a closing whole pass.
    pub epoch_rows: Vec<usize>,
    pub total_steps: usize,
    /// Revival threshold as a fraction of the mean mass; 0 disables it.
    pub revive: f64,
}

/// Round a positive value to the nearest power of two.
fn round_pow2(v: f64) -> usize {
    if v <= 1.0 {
        return 1;
    }
    let lo = 2f64.powi(v.log2().floor() as i32);
    let hi = 2.0 * lo;
    (if v / lo < hi / v { lo } else { hi }) as usize
}

/// Whole passes, a rounded fractional prefix, then a closing whole pass.
/// The prefix rounds half to even, as Python's `round` does.
fn epoch_rows(n: usize, epochs: f64) -> Vec<usize> {
    let whole = epochs.floor();
    let remainder = ((epochs - whole) * n as f64).round_ties_even() as usize;
    let whole = whole as usize;
    if whole == 0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Resolve the batch size, initial step, epoch plan and revival threshold
/// for `n` rows of dimension `d` and `k` clusters over `epochs` passes. A
/// given `batch_size` overrides the batch law, and a given `alpha0` the
/// step law, which otherwise follows the batch size actually used.
#[allow(clippy::too_many_arguments)]
pub fn recipe(
    n: usize,
    k: usize,
    d: usize,
    epochs: f64,
    law: BatchLaw,
    batch_size: Option<usize>,
    alpha0: Option<f64>,
    revive: Option<f64>,
) -> Recipe {
    let resolved = match law {
        BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
        BatchLaw::Auto => BatchLaw::Throughput,
        given => given,
    };
    let (nf, kf) = (n as f64, k as f64);
    let (law, batch_size) = match batch_size {
        Some(b) => (None, b.min(n)),
        None => {
            let [c, bk, bn, bt] =
                if resolved == BatchLaw::Quality { QUALITY_LAW } else { THROUGHPUT_LAW };
            let b = round_pow2(c * kf.powf(bk) * nf.powf(bn) * epochs.powf(bt));
            // The two caps differ deliberately: they reproduce the two
            // fitted recipes.
            let b = if resolved == BatchLaw::Throughput {
                b.min(round_pow2(nf / 2.0))
            } else {
                b.min(n / 2)
            };
            (Some(resolved), b.min(n).max(1))
        }
    };
    let alpha0 = alpha0.unwrap_or_else(|| {
        let [c, ak, ab, an, at] = ALPHA_LAW;
        let a = c * kf.powf(ak) * (batch_size as f64).powf(ab) * nf.powf(an) * epochs.powf(at);
        a.clamp(1e-4, 1.0)
    });
    let epoch_rows = epoch_rows(n, epochs);
    let total_steps = epoch_rows.iter().map(|r| r.div_ceil(batch_size)).sum();
    let revive = revive.unwrap_or(if d <= NO_REVIVAL_MAX_D { 0.0 } else { UNUSED_THRESHOLD });
    Recipe { law, batch_size, alpha0, epoch_rows, total_steps, revive }
}

/// Cosine-anneal the step from `alpha0` towards a tenth of it.
fn scheduled_alpha(alpha0: f64, step: usize, total: usize) -> f64 {
    let fraction = step.min(total) as f64 / total.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// A seeded stream of uniform draws.
struct Rng(Xoshiro256PlusPlus);

impl Rng {
    /// Uniform on `[0, 1)` from the top 53 bits.
    fn uniform(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on `0..n`, unbiased (Lemire's multiply-and-reject).
    fn below(&mut self, n: usize) -> usize {
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = self.0.next_u64() as u128 * n as u128;
            if m as u64 >= threshold {
                return (m >> 64) as usize;
            }
        }
    }

    /// Make the first `m` entries of `v` a uniform random sample of its
    /// entries, in random order (a partial Fisher-Yates shuffle).
    fn shuffle_prefix(&mut self, v: &mut [usize], m: usize) {
        let n = v.len();
        for i in 0..m.min(n) {
            let j = i + self.below(n - i);
            v.swap(i, j);
        }
    }
}

/// Independent streams for the start, the epoch orders and revival, from
/// one seed by xoshiro's jump function.
fn streams(seed: u64) -> (Rng, Rng, Rng) {
    let mut base = Xoshiro256PlusPlus::seed_from_u64(seed);
    let init = base.clone();
    base.jump();
    let order = base.clone();
    base.jump();
    (Rng(init), Rng(order), Rng(base))
}

/// Scale `v` to unit length, prescaling by its largest magnitude so the
/// squared norm cannot overflow. Zero stays zero.
fn normalise(v: &mut [f64]) {
    let max = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if max == 0.0 {
        return;
    }
    let norm = v.iter().map(|x| (x / max) * (x / max)).sum::<f64>().sqrt();
    for x in v.iter_mut() {
        *x = (*x / max) / norm;
    }
}

/// Normalise every row of row-major `x` in place.
fn normalise_rows(x: &mut [f64], d: usize) {
    x.par_chunks_mut(d).for_each(normalise);
}

/// Rows of `x` with every coordinate zero, which have no direction.
pub fn count_zero_rows(x: &[f64], d: usize) -> usize {
    x.par_chunks(d).filter(|row| row.iter().all(|&v| v == 0.0)).count()
}

/// Column means of row-major `x`, summed in fixed blocks and the blocks in
/// order, so the result does not depend on the thread count.
fn column_means(x: &[f64], d: usize) -> Vec<f64> {
    let n = x.len() / d;
    let partial: Vec<Vec<f64>> = x
        .par_chunks(MEAN_BLOCK * d)
        .map(|block| {
            let mut s = vec![0.0; d];
            for row in block.chunks_exact(d) {
                for (a, v) in s.iter_mut().zip(row) {
                    *a += v;
                }
            }
            s
        })
        .collect();
    let mut mean = vec![0.0; d];
    for s in &partial {
        for (m, v) in mean.iter_mut().zip(s) {
            *m += v;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    mean
}

/// Squared norm of each row less `shift`.
fn squared_norms(x: &[f64], d: usize, shift: Option<&[f64]>) -> Vec<f64> {
    x.par_chunks(d)
        .map(|row| match shift {
            Some(s) => row.iter().zip(s).map(|(v, m)| (v - m) * (v - m)).sum(),
            None => row.iter().map(|v| v * v).sum(),
        })
        .collect()
}

/// The rows an assignment covers: all of `x` in order, or the listed rows.
#[derive(Clone, Copy)]
enum Rows<'a> {
    All,
    Listed(&'a [usize]),
}

/// Nearest centroid of each covered row less `shift`, as the argmin over
/// `j` of `cn[j] - 2 x.c_j`. With `cn = |c|^2` that is the nearest centroid
/// in Euclidean distance; with `cn = 0` it is the largest inner product.
/// Writes the label and the minimised score per row, ties to the lowest
/// index. Tiles are fixed by row position, so the result does not depend on
/// the thread count; the rows of a tile are gathered (and shifted) into a
/// buffer, so it does not depend on where a row sits either.
#[allow(clippy::too_many_arguments)]
fn nearest(
    x: &[f64],
    d: usize,
    rows: Rows,
    shift: Option<&[f64]>,
    c: &[f64],
    cn: &[f64],
    labels: &mut [u32],
    best: &mut [f64],
) {
    let k = cn.len();
    let width = COL_BLOCK.min(k);
    labels
        .par_chunks_mut(ROW_BLOCK)
        .zip(best.par_chunks_mut(ROW_BLOCK))
        .enumerate()
        .for_each_init(
            || (Vec::<f64>::with_capacity(ROW_BLOCK * d), Array2::<f64>::zeros((ROW_BLOCK, width))),
            |(gathered, product), (t, (lab, bst))| {
                let m = lab.len();
                let r0 = t * ROW_BLOCK;
                gathered.clear();
                for i in 0..m {
                    let r = match rows {
                        Rows::All => r0 + i,
                        Rows::Listed(idx) => idx[r0 + i],
                    };
                    let row = &x[r * d..(r + 1) * d];
                    match shift {
                        Some(s) => gathered.extend(row.iter().zip(s).map(|(v, mu)| v - mu)),
                        None => gathered.extend_from_slice(row),
                    }
                }
                let a = ArrayView2::from_shape((m, d), &gathered[..])
                    .expect("tile shape matches its buffer");
                lab.fill(0);
                bst.fill(f64::INFINITY);
                for j0 in (0..k).step_by(COL_BLOCK) {
                    let j1 = (j0 + COL_BLOCK).min(k);
                    let cb = ArrayView2::from_shape((j1 - j0, d), &c[j0 * d..j1 * d])
                        .expect("centroid block shape matches its slice");
                    let mut g = product.slice_mut(s![..m, ..j1 - j0]);
                    general_mat_mul(1.0, &a, &cb.t(), 0.0, &mut g);
                    for (i, row) in g.rows().into_iter().enumerate() {
                        let row = row.as_slice().expect("product rows are contiguous");
                        let (mut bi, mut bs) = (lab[i], bst[i]);
                        for (jj, (&dot, &norm)) in row.iter().zip(&cn[j0..j1]).enumerate() {
                            let score = norm - 2.0 * dot;
                            if score < bs {
                                bs = score;
                                bi = (j0 + jj) as u32;
                            }
                        }
                        lab[i] = bi;
                        bst[i] = bs;
                    }
                }
            },
        );
}

/// Per-centroid counts and sums of the listed rows less `shift`. Each task
/// owns a range of centroids and scans the batch in row order, so every sum
/// is formed in the same order whatever the thread count.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    x: &[f64],
    d: usize,
    idx: &[usize],
    labels: &[u32],
    shift: Option<&[f64]>,
    sums: &mut [f64],
    counts: &mut [f64],
) {
    let k = counts.len();
    let per_task = k.div_ceil(ACCUMULATE_TASKS).max(1);
    sums.par_chunks_mut(per_task * d)
        .zip(counts.par_chunks_mut(per_task))
        .enumerate()
        .for_each(|(t, (s, c))| {
            s.fill(0.0);
            c.fill(0.0);
            let j0 = t * per_task;
            let j1 = j0 + c.len();
            for (&r, &j) in idx.iter().zip(labels) {
                let j = j as usize;
                if j < j0 || j >= j1 {
                    continue;
                }
                let local = j - j0;
                c[local] += 1.0;
                let row = &x[r * d..(r + 1) * d];
                let acc = &mut s[local * d..(local + 1) * d];
                match shift {
                    Some(m) => {
                        for ((a, v), mu) in acc.iter_mut().zip(row).zip(m) {
                            *a += v - mu;
                        }
                    }
                    None => {
                        for (a, v) in acc.iter_mut().zip(row) {
                            *a += v;
                        }
                    }
                }
            }
        });
}

/// The centroids assignment searches: the stored positions, or in cosine
/// mode their unit directions (written into `buf`). Also fills `cn` with
/// the norm term [`nearest`] needs.
fn readout<'a>(c: &'a [f64], d: usize, cosine: bool, buf: &'a mut [f64], cn: &mut [f64]) -> &'a [f64] {
    if cosine {
        buf.copy_from_slice(c);
        buf.chunks_exact_mut(d).for_each(normalise);
        cn.fill(0.0);
        buf
    } else {
        for (n, row) in cn.iter_mut().zip(c.chunks_exact(d)) {
            *n = row.iter().map(|v| v * v).sum();
        }
        c
    }
}

/// Apply one batch's counts and sums to the masses and positions
/// (Algorithm 1, line 8). Centroids without points keep their positions;
/// every mass decays unless the step is 1, when occupied masses are
/// replaced outright.
fn apply_batch(c: &mut [f64], usage: &mut [f64], sums: &[f64], counts: &[f64], d: usize, alpha: f64) {
    let replace = alpha == 1.0;
    if !replace {
        let decay = 1.0 - alpha;
        usage.iter_mut().for_each(|u| *u *= decay);
    }
    for (j, &count) in counts.iter().enumerate() {
        if count == 0.0 {
            continue;
        }
        let kept = if replace { 0.0 } else { usage[j] };
        let mass = kept + alpha * count;
        let weight = alpha * count / mass;
        let (pos, sum) = (&mut c[j * d..(j + 1) * d], &sums[j * d..(j + 1) * d]);
        for (p, &s) in pos.iter_mut().zip(sum) {
            let mean = s / count;
            *p = if weight == 1.0 { mean } else { *p + weight * (mean - *p) };
        }
        usage[j] = mass;
    }
}

/// The next representable value above `v`, for finite `v`.
fn next_up(v: f64) -> f64 {
    if v == 0.0 {
        return f64::from_bits(1);
    }
    let bits = v.to_bits();
    f64::from_bits(if v > 0.0 { bits + 1 } else { bits - 1 })
}

/// Revive centroids whose mass is below `tau` times the mean mass, by
/// splitting donors drawn without replacement in proportion to their mass
/// (Gumbel top-k on `uniform` draws, one per centroid). Each donor and its
/// child are the donor's position nudged apart by [`SPLIT_EPSILON`] in
/// alternating directions, and share its mass. Returns the number of splits.
fn revive(c: &mut [f64], usage: &mut [f64], d: usize, tau: f64, uniform: &mut dyn FnMut() -> f64) -> usize {
    let k = usage.len();
    let threshold = usage.iter().sum::<f64>() * tau / k as f64;
    let targets: Vec<usize> = (0..k).filter(|&j| usage[j] < threshold).collect();
    let donors: Vec<usize> = (0..k).filter(|&j| usage[j] >= threshold && usage[j] > 0.0).collect();
    let n_split = targets.len().min(donors.len());
    if n_split == 0 {
        return 0;
    }
    // One draw per centroid in index order, as the reference draws them.
    let u: Vec<f64> = (0..k).map(|_| uniform().max(f64::MIN_POSITIVE)).collect();
    let mut keyed: Vec<(f64, usize)> =
        donors.iter().map(|&j| (usage[j].ln() - (-u[j].ln()).ln(), j)).collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut donor = vec![0.0; d];
    let mut child = vec![0.0; d];
    for (&(_, src), &dst) in keyed.iter().zip(&targets).take(n_split) {
        let base = &c[src * d..(src + 1) * d];
        for i in 0..d {
            let offset = if i % 2 == 0 { SPLIT_EPSILON } else { -SPLIT_EPSILON };
            donor[i] = base[i] * (1.0 - offset);
            child[i] = base[i] * (1.0 + offset);
        }
        // A zero position perturbs to itself; separate the two along the
        // first coordinate.
        if donor == child {
            let up = next_up(child[0]);
            child[0] = if up.is_finite() { up } else { -next_up(-child[0]) };
        }
        let half = usage[src] / 2.0;
        usage[src] = half;
        usage[dst] = half;
        c[src * d..(src + 1) * d].copy_from_slice(&donor);
        c[dst * d..(dst + 1) * d].copy_from_slice(&child);
    }
    n_split
}

/// The trained centroids and the record of the run.
struct Trained {
    centroids: Vec<f64>,
    /// Per epoch, the sum over its rows of the squared distance to the
    /// centroids each row was assigned against (they move during the epoch).
    epoch_objective: Vec<f64>,
    n_splits: usize,
}

/// Train from `init` (`k x d`, row-major, in coordinates less `shift`)
/// under `recipe`, returning centroids in the same coordinates. Rows are
/// read as `x` less `shift`; `x` must already be normalised in cosine mode,
/// where `shift` is `None`. `order(epoch, v)` arranges the row indices `v`
/// for an epoch, of which the first `recipe.epoch_rows[epoch]` are visited,
/// and `uniform` supplies revival's draws.
#[allow(clippy::too_many_arguments)]
fn train(
    x: &[f64],
    d: usize,
    shift: Option<&[f64]>,
    mut c: Vec<f64>,
    recipe: &Recipe,
    cosine: bool,
    mut order: impl FnMut(usize, &mut [usize]),
    uniform: &mut dyn FnMut() -> f64,
) -> Trained {
    let n = x.len() / d;
    let k = c.len() / d;
    if cosine {
        c.chunks_exact_mut(d).for_each(normalise);
    }
    let norms = if cosine { Vec::new() } else { squared_norms(x, d, shift) };
    let b = recipe.batch_size;
    let last = recipe.epoch_rows.len() - 1;

    let mut usage = vec![INIT_USAGE; k];
    let mut perm: Vec<usize> = (0..n).collect();
    let mut labels = vec![0u32; b];
    let mut best = vec![0.0; b];
    let mut sums = vec![0.0; k * d];
    let mut counts = vec![0.0; k];
    let mut directions = vec![0.0; if cosine { k * d } else { 0 }];
    let mut cn = vec![0.0; k];
    let mut pass_counts = vec![0.0; k];
    let mut pass_sums = vec![0.0; k * d];
    let mut epoch_objective = Vec::with_capacity(recipe.epoch_rows.len());
    let (mut step, mut n_splits) = (0usize, 0usize);

    for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
        order(epoch, &mut perm);
        pass_counts.fill(0.0);
        let mut objective = 0.0;
        for b0 in (0..rows).step_by(b) {
            let idx = &perm[b0..(b0 + b).min(rows)];
            let m = idx.len();
            let search = readout(&c, d, cosine, &mut directions, &mut cn);
            nearest(x, d, Rows::Listed(idx), shift, search, &cn, &mut labels[..m], &mut best[..m]);
            accumulate(x, d, idx, &labels[..m], shift, &mut sums, &mut counts);

            // Squared distance: |x|^2 + |c|^2 - 2 x.c, or for unit vectors
            // 2 - 2 x.c. Summed in row order.
            for (&r, &score) in idx.iter().zip(&best[..m]) {
                objective += if cosine { 2.0 + score } else { (norms[r] + score).max(0.0) };
            }

            let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps);
            for (p, q) in pass_counts.iter_mut().zip(&counts) {
                *p += q;
            }
            if epoch == last {
                for (p, q) in pass_sums.iter_mut().zip(&sums) {
                    *p += q;
                }
            }
            apply_batch(&mut c, &mut usage, &sums, &counts, d, alpha);
            if (step + 1) % CHECK_UNUSED_EVERY == 0 && recipe.revive > 0.0 {
                n_splits += revive(&mut c, &mut usage, d, recipe.revive, uniform);
            }
            step += 1;
        }
        epoch_objective.push(objective);
    }

    // Closing means over the final epoch (Algorithm 1, line 12); centroids
    // it never assigned to keep their positions.
    for (j, &count) in pass_counts.iter().enumerate() {
        if count > 0.0 {
            for (p, &s) in c[j * d..(j + 1) * d].iter_mut().zip(&pass_sums[j * d..(j + 1) * d]) {
                *p = s / count;
            }
        }
    }
    if cosine {
        c.chunks_exact_mut(d).for_each(normalise);
    }
    Trained { centroids: c, epoch_objective, n_splits }
}

pub struct Params {
    pub k: usize,
    pub epochs: f64,
    pub law: BatchLaw,
    pub batch_size: Option<usize>,
    pub alpha0: Option<f64>,
    pub revive: Option<f64>,
    pub cosine: bool,
    pub seed: u64,
}

pub struct Fit {
    /// Zero-based nearest final centroid per row.
    pub labels: Vec<u32>,
    /// `k x d`, row-major.
    pub centroids: Vec<f64>,
    /// Sum of squared distances to the assigned centroid; in cosine mode,
    /// between the normalised rows and the unit centroids.
    pub inertia: f64,
    /// False when the distances overflowed, so the labels mean nothing.
    pub finite: bool,
    pub sizes: Vec<i32>,
    pub epoch_objective: Vec<f64>,
    pub n_splits: usize,
    pub recipe: Recipe,
}

/// Exact assignment of every row and the inertia of that partition, with
/// whether every distance was finite. Euclidean distances are computed
/// relative to the mean of the centroids, which a saved model can recompute,
/// so [`assign`] on the training rows reproduces the labels of [`fit`]. The
/// inertia is summed per tile and the tiles in order.
fn assign_all(x: &[f64], d: usize, c: &[f64], cosine: bool) -> (Vec<u32>, f64, bool) {
    let n = x.len() / d;
    let k = c.len() / d;
    let shift: Option<Vec<f64>> = (!cosine).then(|| {
        let mut m = vec![0.0; d];
        for row in c.chunks_exact(d) {
            for (a, v) in m.iter_mut().zip(row) {
                *a += v;
            }
        }
        m.iter_mut().for_each(|a| *a /= k as f64);
        m
    });
    let cs: Vec<f64> = match &shift {
        Some(m) => c.chunks_exact(d).flat_map(|row| row.iter().zip(m).map(|(v, mu)| v - mu)).collect(),
        None => c.to_vec(),
    };
    let cn: Vec<f64> = if cosine { vec![0.0; k] } else { squared_norms(&cs, d, None) };
    let mut labels = vec![0u32; n];
    let mut best = vec![0.0; n];
    nearest(x, d, Rows::All, shift.as_deref(), &cs, &cn, &mut labels, &mut best);
    let finite = best.par_iter().all(|v| v.is_finite());
    let partial: Vec<f64> = x
        .par_chunks(ROW_BLOCK * d)
        .zip(labels.par_chunks(ROW_BLOCK))
        .map(|(xs, ls)| {
            xs.chunks_exact(d)
                .zip(ls)
                .map(|(row, &j)| {
                    let cj = &cs[j as usize * d..(j as usize + 1) * d];
                    match &shift {
                        Some(m) => row
                            .iter()
                            .zip(m)
                            .zip(cj)
                            .map(|((v, mu), b)| (v - mu - b) * (v - mu - b))
                            .sum::<f64>(),
                        None => row.iter().zip(cj).map(|(a, b)| (a - b) * (a - b)).sum::<f64>(),
                    }
                })
                .sum()
        })
        .collect();
    let inertia: f64 = partial.iter().sum();
    (labels, inertia, finite && inertia.is_finite())
}

/// Fit e-means to row-major `x` (`n x d`), with `1 <= k <= n`. In cosine
/// mode the rows are normalised first and must not be zero. Runs on the
/// current pool.
pub fn fit(mut x: Vec<f64>, d: usize, p: &Params) -> Fit {
    let n = x.len() / d;
    assert!(p.k >= 1 && p.k <= n, "e-means needs 1 <= k <= n, got k = {} and n = {n}", p.k);
    if p.cosine {
        normalise_rows(&mut x, d);
    }
    let recipe = recipe(n, p.k, d, p.epochs, p.law, p.batch_size, p.alpha0, p.revive);
    let (mut init_rng, mut order_rng, mut split_rng) = streams(p.seed);
    let shift: Option<Vec<f64>> = (!p.cosine).then(|| column_means(&x, d));

    // The start is k distinct rows drawn uniformly, or every row when k = n,
    // in the training coordinates.
    let rows: Vec<usize> = if p.k == n {
        (0..n).collect()
    } else {
        let mut pick: Vec<usize> = (0..n).collect();
        init_rng.shuffle_prefix(&mut pick, p.k);
        pick.truncate(p.k);
        pick
    };
    let mean = shift.as_deref();
    let init: Vec<f64> = rows
        .iter()
        .flat_map(|&r| {
            let row = &x[r * d..(r + 1) * d];
            (0..d).map(move |t| row[t] - mean.map_or(0.0, |m| m[t]))
        })
        .collect();

    let order = |epoch: usize, v: &mut [usize]| order_rng.shuffle_prefix(v, recipe.epoch_rows[epoch]);
    let mut draw = || split_rng.uniform();
    let trained = train(&x, d, mean, init, &recipe, p.cosine, order, &mut draw);

    let mut centroids = trained.centroids;
    if let Some(m) = &shift {
        for row in centroids.chunks_exact_mut(d) {
            for (v, mu) in row.iter_mut().zip(m) {
                *v += mu;
            }
        }
    }
    let (labels, inertia, finite) = assign_all(&x, d, &centroids, p.cosine);
    let mut sizes = vec![0i32; p.k];
    for &j in &labels {
        sizes[j as usize] += 1;
    }
    Fit {
        labels,
        finite: finite && centroids.iter().all(|v| v.is_finite()),
        centroids,
        inertia,
        sizes,
        epoch_objective: trained.epoch_objective,
        n_splits: trained.n_splits,
        recipe,
    }
}

/// Nearest centroid of each row of `x` (`n x d`, row-major), normalising
/// the rows first in cosine mode, and whether every distance was finite.
/// Backs `predict()`, and matches the final assignment of [`fit`] row for
/// row. Runs on the current pool.
pub fn assign(mut x: Vec<f64>, d: usize, centroids: &[f64], cosine: bool) -> (Vec<u32>, bool) {
    assert!(!centroids.is_empty() && centroids.len() % d == 0, "centroids must be k x d with k >= 1");
    if cosine {
        normalise_rows(&mut x, d);
    }
    let (labels, _, finite) = assign_all(&x, d, centroids, cosine);
    (labels, finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emeans_fixtures::{RECIPES, RUNS};

    /// The fixture data: four loose groups, rounded to single precision as
    /// the reference saw it.
    fn fixture_data(n: usize, d: usize) -> Vec<f64> {
        let mut x = Vec::with_capacity(n * d);
        for i in 0..n {
            let g = (i % 4) as f64;
            for j in 0..d {
                let j = j as f64;
                let v = 1.5 * (g * 1.7 + j * 0.9).cos() + 0.3 * (0.37 * i as f64 + 1.3 * j).sin();
                x.push(v as f32 as f64);
            }
        }
        x
    }

    fn cloud(n: usize, d: usize, seed: f64) -> Vec<f64> {
        (0..n * d).map(|i| (i as f64 * 0.37 + seed).sin() + ((i / d) % 5) as f64).collect()
    }

    fn params(k: usize, cosine: bool, seed: u64) -> Params {
        Params {
            k,
            epochs: 3.0,
            law: BatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            revive: None,
            cosine,
            seed,
        }
    }

    fn brute_nearest(x: &[f64], c: &[f64], d: usize) -> Vec<u32> {
        let k = c.len() / d;
        x.chunks_exact(d)
            .map(|row| {
                let dist = |j: usize| -> f64 {
                    row.iter().zip(&c[j * d..(j + 1) * d]).map(|(a, b)| (a - b) * (a - b)).sum()
                };
                (0..k).min_by(|&a, &b| dist(a).total_cmp(&dist(b))).unwrap() as u32
            })
            .collect()
    }

    #[test]
    fn recipes_match_the_reference() {
        for case in RECIPES {
            let r = recipe(
                case.n,
                case.k,
                case.d,
                case.epochs,
                BatchLaw::from_name(case.law),
                case.batch_size,
                case.alpha0,
                case.revive,
            );
            let at = format!("n {} k {} d {} epochs {}", case.n, case.k, case.d, case.epochs);
            assert_eq!(r.law.map_or("", |l| l.name()), case.want_law, "{at}");
            assert_eq!(r.batch_size, case.want_batch, "{at}");
            assert!((r.alpha0 - case.want_alpha0).abs() <= 1e-12 * case.want_alpha0, "{at}: {}", r.alpha0);
            assert_eq!(r.epoch_rows, case.want_rows, "{at}");
            assert_eq!(r.total_steps, case.want_steps, "{at}");
            assert_eq!(r.revive, case.want_revive, "{at}");
        }
    }

    #[test]
    fn runs_match_the_reference() {
        for case in RUNS {
            let mut x = fixture_data(case.n, case.d);
            if case.cosine {
                normalise_rows(&mut x, case.d);
            }
            let mut init: Vec<f64> =
                case.init_rows.iter().flat_map(|&r| x[r * case.d..(r + 1) * case.d].to_vec()).collect();
            init.extend_from_slice(case.init_extra);
            assert_eq!(init.len(), case.k * case.d, "{}", case.name);
            let rows = epoch_rows(case.n, case.epochs);
            let recipe = Recipe {
                law: None,
                batch_size: case.batch,
                alpha0: case.alpha0,
                total_steps: rows.iter().map(|r| r.div_ceil(case.batch)).sum(),
                epoch_rows: rows,
                revive: case.revive,
            };
            // Rows in their natural order every epoch, and revival drawing
            // the reference's own uniforms, as the fixtures were generated.
            let mut draws = case.uniforms.iter();
            let mut uniform = || *draws.next().expect("the run used more draws than the reference");
            let t = train(&x, case.d, None, init, &recipe, case.cosine, |_, _| {}, &mut uniform);

            assert_eq!(t.n_splits, case.want_splits, "{}", case.name);
            assert_eq!(t.centroids.len(), case.want_centroids.len(), "{}", case.name);
            for (got, want) in t.centroids.iter().zip(case.want_centroids) {
                assert!((got - want).abs() < 1e-5, "{}: centroid {got} vs {want}", case.name);
            }
            assert_eq!(t.epoch_objective.len(), case.want_objective.len(), "{}", case.name);
            for (got, want) in t.epoch_objective.iter().zip(case.want_objective) {
                assert!((got - want).abs() < 1e-5 * want.abs().max(1.0), "{}: objective {got} vs {want}", case.name);
            }
        }
    }

    #[test]
    fn nearest_agrees_with_brute_force_across_tiles() {
        // Rows span three row tiles and centroids two column tiles, both
        // ragged; listed rows exercise the gather path, and a shift the
        // centring.
        let (n, k, d) = (2 * ROW_BLOCK + 5, COL_BLOCK + 37, 5);
        let x = cloud(n, d, 0.1);
        let c = cloud(k, d, 0.9);
        let brute = brute_nearest(&x, &c, d);
        let cn = squared_norms(&c, d, None);

        let (mut lab, mut best) = (vec![0u32; n], vec![0.0; n]);
        nearest(&x, d, Rows::All, None, &c, &cn, &mut lab, &mut best);
        assert_eq!(lab, brute);

        let listed: Vec<usize> = (0..n).rev().collect();
        nearest(&x, d, Rows::Listed(&listed), None, &c, &cn, &mut lab, &mut best);
        let expected: Vec<u32> = listed.iter().map(|&r| brute[r]).collect();
        assert_eq!(lab, expected);

        let shift = vec![0.5, -1.0, 2.0, 0.0, 3.0];
        let cs: Vec<f64> =
            c.chunks_exact(d).flat_map(|r| r.iter().zip(&shift).map(|(v, m)| v - m).collect::<Vec<_>>()).collect();
        let csn = squared_norms(&cs, d, None);
        nearest(&x, d, Rows::All, Some(&shift), &cs, &csn, &mut lab, &mut best);
        assert_eq!(lab, brute);
    }

    #[test]
    fn accumulate_matches_a_serial_sum() {
        let (n, k, d) = (500, 200, 3);
        let x = cloud(n, d, 0.2);
        let idx: Vec<usize> = (0..n).rev().step_by(2).collect();
        let labels: Vec<u32> = idx.iter().map(|&r| ((r * 7) % k) as u32).collect();
        let shift = vec![0.1, 0.2, 0.3];
        let (mut sums, mut counts) = (vec![1.0; k * d], vec![1.0; k]);
        accumulate(&x, d, &idx, &labels, Some(&shift), &mut sums, &mut counts);

        let (mut want_s, mut want_c) = (vec![0.0; k * d], vec![0.0; k]);
        for (&r, &j) in idx.iter().zip(&labels) {
            let j = j as usize;
            want_c[j] += 1.0;
            for t in 0..d {
                want_s[j * d + t] += x[r * d + t] - shift[t];
            }
        }
        assert_eq!(counts, want_c);
        assert_eq!(sums, want_s);
    }

    #[test]
    fn apply_batch_follows_the_update_rule() {
        let d = 2;
        let sums = [4.0, 6.0, 0.0, 0.0, 9.0, 3.0];
        let counts = [2.0, 0.0, 3.0];

        // A step of 1 replaces occupied masses and positions outright, and
        // leaves an empty centroid and its mass alone.
        let (mut c, mut usage) = (vec![10.0, 10.0, -1.0, -1.0, 0.0, 0.0], vec![5.0, 0.5, 7.0]);
        apply_batch(&mut c, &mut usage, &sums, &counts, d, 1.0);
        assert_eq!(c, [2.0, 3.0, -1.0, -1.0, 3.0, 1.0]);
        assert_eq!(usage, [2.0, 0.5, 3.0]);

        // Below 1 every mass decays, and an occupied centroid moves towards
        // its batch mean by alpha n / N.
        let (mut c, mut usage) = (vec![10.0, 10.0, -1.0, -1.0, 0.0, 0.0], vec![5.0, 0.5, 7.0]);
        apply_batch(&mut c, &mut usage, &sums, &counts, d, 0.25);
        let mass0 = 5.0 * 0.75 + 0.25 * 2.0;
        let w0 = 0.25 * 2.0 / mass0;
        assert_eq!(usage[0], mass0);
        assert_eq!(usage[1], 0.5 * 0.75);
        assert!((c[0] - (10.0 + w0 * (2.0 - 10.0))).abs() < 1e-15);
        assert_eq!(&c[2..4], [-1.0, -1.0]);
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        // Batches of several tiles, and more centroids than one column tile
        // and than one accumulation task, so every parallel stage is split.
        let x = cloud(3000, 6, 0.3);
        let run = |threads: usize, cosine: bool| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut p = params(1100, cosine, 7);
            p.batch_size = Some(512);
            pool.install(|| fit(x.clone(), 6, &p))
        };
        for cosine in [false, true] {
            let one = run(1, cosine);
            for threads in [3, 4] {
                let other = run(threads, cosine);
                assert_eq!(one.labels, other.labels);
                assert_eq!(one.centroids, other.centroids);
                assert_eq!(one.inertia.to_bits(), other.inertia.to_bits());
                assert_eq!(one.epoch_objective, other.epoch_objective);
            }
        }
    }

    #[test]
    fn the_seed_reproduces_and_varies_the_run() {
        let x = cloud(2000, 4, 0.2);
        let a = fit(x.clone(), 4, &params(25, false, 1));
        let b = fit(x.clone(), 4, &params(25, false, 1));
        let c = fit(x, 4, &params(25, false, 2));
        assert_eq!(a.centroids, b.centroids);
        assert_ne!(a.centroids, c.centroids);
    }

    #[test]
    fn the_fit_describes_one_partition() {
        let x = cloud(1500, 3, 0.4);
        let f = fit(x.clone(), 3, &params(12, false, 3));
        assert!(f.finite);
        assert_eq!(f.sizes.iter().sum::<i32>(), 1500);
        assert_eq!(assign(x.clone(), 3, &f.centroids, false).0, f.labels);
        assert_eq!(brute_nearest(&x, &f.centroids, 3), f.labels);
        let wss: f64 = x
            .chunks_exact(3)
            .zip(&f.labels)
            .map(|(r, &j)| r.iter().zip(&f.centroids[j as usize * 3..]).map(|(a, b)| (a - b) * (a - b)).sum::<f64>())
            .sum();
        assert!((f.inertia - wss).abs() <= 1e-9 * wss);
        assert_eq!(f.epoch_objective.len(), 3);
    }

    #[test]
    fn where_the_data_sits_does_not_matter() {
        // Projected coordinates sit far from the origin; centring makes the
        // fit the same as on local coordinates, up to the rounding of the
        // offset itself.
        let x = cloud(4000, 3, 0.8);
        let far: Vec<f64> = x.iter().enumerate().map(|(i, v)| v + [4.5e5, 5.6e6, 300.0][i % 3]).collect();
        let near = fit(x, 3, &params(64, false, 5));
        let shifted = fit(far, 3, &params(64, false, 5));
        assert_eq!(near.labels, shifted.labels);
        assert_eq!(near.n_splits, shifted.n_splits);
        assert!((near.inertia - shifted.inertia).abs() <= 1e-6 * near.inertia);
    }

    #[test]
    fn overflow_is_reported() {
        let x: Vec<f64> = cloud(200, 3, 0.1).iter().map(|v| v * 1e160).collect();
        assert!(!fit(x.clone(), 3, &params(4, false, 1)).finite);
        let c = vec![1e160; 6];
        assert!(!assign(x, 3, &c, false).1);
    }

    #[test]
    fn cosine_mode_ignores_row_scale() {
        let x = cloud(1200, 5, 0.6);
        // Powers of two scale exactly, so normalisation sees identical rows.
        let scaled: Vec<f64> = x
            .chunks_exact(5)
            .enumerate()
            .flat_map(|(i, r)| r.iter().map(move |v| v * [0.25, 1.0, 8.0][i % 3]))
            .collect();
        let a = fit(x, 5, &params(10, true, 4));
        let b = fit(scaled, 5, &params(10, true, 4));
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.centroids, b.centroids);
        for row in a.centroids.chunks_exact(5) {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn edge_sizes_run() {
        let x = cloud(30, 3, 0.5);
        // k = n starts from every row; k = 1 is the mean; a fraction of an
        // epoch visits only a prefix.
        let all = fit(x.clone(), 3, &params(30, false, 1));
        assert_eq!(all.sizes.len(), 30);
        let one = fit(x.clone(), 3, &params(1, false, 1));
        assert!(one.labels.iter().all(|&j| j == 0));
        let mut short = params(4, false, 1);
        short.epochs = 0.5;
        let f = fit(x.clone(), 3, &short);
        assert_eq!(f.recipe.epoch_rows, vec![15]);
        assert_eq!(f.epoch_objective.len(), 1);
        let single = fit(x[..3].to_vec(), 3, &params(1, true, 1));
        assert_eq!(single.labels, [0]);
    }

    #[test]
    #[should_panic(expected = "1 <= k <= n")]
    fn k_beyond_n_is_refused() {
        fit(cloud(5, 2, 0.1), 2, &params(6, false, 1));
    }

    #[test]
    fn revival_splits_a_starved_centroid() {
        // A start far from every point never wins one, so its mass decays
        // until it is revived by splitting the other.
        let x = cloud(400, 3, 0.7);
        let init = [x[..3].to_vec(), vec![1e3; 3]].concat();
        let r = recipe(400, 2, 3, 2.0, BatchLaw::Throughput, Some(20), Some(0.5), Some(0.1));
        let (_, _, mut split) = streams(0);
        let mut uniform = || split.uniform();
        let t = train(&x, 3, None, init.clone(), &r, false, |_, _| {}, &mut uniform);
        assert!(t.n_splits >= 1);
        assert!(t.centroids[3..].iter().all(|v| v.abs() < 10.0), "{:?}", t.centroids);
        assert_ne!(t.centroids[..3], t.centroids[3..]);

        // With revival off it stays where it started.
        let mut off = r.clone();
        off.revive = 0.0;
        let t = train(&x, 3, None, init, &off, false, |_, _| {}, &mut uniform);
        assert_eq!(t.n_splits, 0);
        assert_eq!(t.centroids[3..], [1e3; 3]);
    }

    #[test]
    fn rng_helpers_are_in_range_and_unbiased_enough() {
        let (mut a, _, _) = streams(9);
        let mut hist = [0usize; 3];
        for _ in 0..30_000 {
            hist[a.below(3)] += 1;
            let u = a.uniform();
            assert!((0.0..1.0).contains(&u));
        }
        assert!(hist.iter().all(|&h| (9_500..10_500).contains(&h)), "{hist:?}");
        let mut v: Vec<usize> = (0..50).collect();
        a.shuffle_prefix(&mut v, 50);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_eq!(next_up(0.0), f64::from_bits(1));
        assert!(next_up(-1.0) > -1.0 && next_up(1.0) > 1.0);
    }
}
