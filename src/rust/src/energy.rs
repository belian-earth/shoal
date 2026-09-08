//! Two-sample energy distance (Szekely and Rizzo).
//!
//! `E(X, Y) = 2 E||x - y|| - E||x - x'|| - E||y - y'||` for independent
//! draws, estimated by the means over all cross pairs and all distinct
//! within-sample pairs. Zero if and only if the two distributions agree,
//! so it is a K-free test of whether two point clouds were drawn from the
//! same distribution, in any dimension.
//!
//! Each of the three means is a dense block of pairwise Euclidean
//! distances that is never materialised: rows are processed in parallel
//! and each row's contribution is reduced to a sum, so the memory is the
//! two inputs and nothing else. Distances are computed as
//! `sqrt(|a|^2 + |b|^2 - 2 a.b)` with the squared norms precomputed, so
//! a row against the whole other sample is one dot product per pair.

use rayon::prelude::*;

pub struct Energy {
    pub cross: f64,
    /// `None` where the caller supplied the term.
    pub self_x: Option<f64>,
    pub self_y: Option<f64>,
}

fn norms(x: &[f64], d: usize) -> Vec<f64> {
    x.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum()).collect()
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for c in 0..a.len() {
        s += a[c] * b[c];
    }
    s
}

/// Mean Euclidean distance over all pairs (rows of `x` against rows of `y`).
fn mean_cross(x: &[f64], nx: &[f64], y: &[f64], ny: &[f64], d: usize) -> f64 {
    let total: f64 = x
        .par_chunks_exact(d)
        .enumerate()
        .map(|(i, a)| {
            let mut s = 0.0;
            for (j, b) in y.chunks_exact(d).enumerate() {
                s += (nx[i] + ny[j] - 2.0 * dot(a, b)).max(0.0).sqrt();
            }
            s
        })
        .sum();
    total / (nx.len() as f64 * ny.len() as f64)
}

/// Mean Euclidean distance over all distinct pairs within `x` (row-major
/// `n x d`). Runs on the current pool.
pub fn self_term(x: &[f64], d: usize) -> f64 {
    let nx = norms(x, d);
    mean_self(x, &nx, d)
}

/// Mean Euclidean distance over all distinct pairs within `x`.
fn mean_self(x: &[f64], nx: &[f64], d: usize) -> f64 {
    let n = nx.len();
    if n < 2 {
        return 0.0;
    }
    // Each unordered pair once: row i against rows j > i.
    let total: f64 = x
        .par_chunks_exact(d)
        .enumerate()
        .map(|(i, a)| {
            let mut s = 0.0;
            for j in (i + 1)..n {
                let b = &x[j * d..(j + 1) * d];
                s += (nx[i] + nx[j] - 2.0 * dot(a, b)).max(0.0).sqrt();
            }
            s
        })
        .sum();
    2.0 * total / (n as f64 * (n as f64 - 1.0))
}

/// `x` row-major `n x d`, `y` row-major `m x d`. The within-sample terms
/// are computed only where requested, so a fixed sample's term can be
/// reused across comparisons. Runs on the current pool.
pub fn energy(x: &[f64], y: &[f64], d: usize, need_x: bool, need_y: bool) -> Energy {
    let nx = norms(x, d);
    let ny = norms(y, d);
    Energy {
        cross: mean_cross(x, &nx, y, &ny, d),
        self_x: need_x.then(|| mean_self(x, &nx, d)),
        self_y: need_y.then(|| mean_self(y, &ny, d)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_have_zero_energy() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin()).collect();
        let e = energy(&x, &x, 2, true, true);
        let (sx, sy) = (e.self_x.unwrap(), e.self_y.unwrap());
        let ed = 2.0 * e.cross - sx - sy;
        // cross includes the zero-distance diagonal, so E is slightly negative
        // for a finite sample compared with itself: -2 self / n.
        assert!((ed + 2.0 * sx / 20.0).abs() < 1e-12, "{ed}");
        assert_eq!(self_term(&x, 2), sx);
        assert!(energy(&x, &x, 2, false, true).self_x.is_none());
    }

    #[test]
    fn shifted_sample_is_further() {
        let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.37).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| v + 0.5).collect();
        let z: Vec<f64> = x.iter().map(|v| v + 2.0).collect();
        let ey = energy(&x, &y, 3, true, true);
        let ez = energy(&x, &z, 3, true, true);
        let ed = |e: &Energy| 2.0 * e.cross - e.self_x.unwrap() - e.self_y.unwrap();
        assert!(ed(&ez) > ed(&ey));
    }
}
