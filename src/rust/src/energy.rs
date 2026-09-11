//! Two-sample energy distance (Szekely and Rizzo).
//!
//! `E(X, Y) = 2 E||x - y|| - E||x - x'|| - E||y - y'||` for independent
//! draws, estimated by the means over all cross pairs and all distinct
//! within-sample pairs. Zero if and only if the two distributions agree,
//! so it is a K-free test of whether two point clouds were drawn from the
//! same distribution, in any dimension.
//!
//! Each of the three means is a dense block of pairwise Euclidean
//! distances, `sqrt(|a|^2 + |b|^2 - 2 a.b)` with the squared norms
//! precomputed. The `a.b` terms are a matrix product, `X Y^T`, so the
//! distance matrix is formed in tiles by gemm (matrixmultiply, through
//! ndarray) and each tile is reduced to a sum as soon as it is formed.
//! A per-pair dot product runs at about one flop per cycle, latency-bound
//! on its accumulator; the gemm runs an order of magnitude faster. Tiles
//! are independent and run in parallel on the package pool, each worker
//! reusing one [`ROW_BLOCK`] x [`COL_BLOCK`] buffer (512 KB), so the
//! memory is the two inputs plus one tile per thread.
//!
//! [`ROW_BLOCK`] is small so that tiles outnumber threads several times
//! over: on a hybrid core layout the wall time is set by the slowest tile,
//! and the gemm loses little at 32 rows.

use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

/// Rows of the first sample per tile.
const ROW_BLOCK: usize = 32;
/// Rows of the second sample per tile.
const COL_BLOCK: usize = 2048;
/// Independent accumulators in the square-root reduction, so it vectorises
/// instead of serialising on one floating-point add.
const LANES: usize = 8;

/// One product tile per worker, reused across the tiles it processes:
/// allocating per tile page-faults a fresh buffer on every worker at once.
fn tile_buffer() -> Array2<f64> {
    Array2::zeros((ROW_BLOCK, COL_BLOCK))
}

pub struct Energy {
    pub cross: f64,
    /// `None` where the caller supplied the term.
    pub self_x: Option<f64>,
    pub self_y: Option<f64>,
}

fn norms(x: &[f64], d: usize) -> Vec<f64> {
    x.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum()).collect()
}

/// Rows `r0..r1` of row-major `x` as a matrix view.
fn rows(x: &[f64], d: usize, r0: usize, r1: usize) -> ArrayView2<'_, f64> {
    ArrayView2::from_shape((r1 - r0, d), &x[r0 * d..r1 * d]).expect("tile shape matches its slice")
}

/// Tiles `(i0, i1, j0, j1)` covering an `n x m` distance matrix. With
/// `upper`, only tiles that meet the strict upper triangle `j > i`.
fn tiles(n: usize, m: usize, upper: bool) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    for i0 in (0..n).step_by(ROW_BLOCK) {
        let i1 = (i0 + ROW_BLOCK).min(n);
        for j0 in (0..m).step_by(COL_BLOCK) {
            let j1 = (j0 + COL_BLOCK).min(m);
            if !upper || j1 > i0 + 1 {
                out.push((i0, i1, j0, j1));
            }
        }
    }
    out
}

/// Sum of the distances in one tile: rows `i0..i1` of `x` against rows
/// `j0..j1` of `y`, restricted to `j > i` when `upper`.
#[allow(clippy::too_many_arguments)]
fn tile_sum(
    buf: &mut Array2<f64>,
    x: &[f64],
    nx: &[f64],
    y: &[f64],
    ny: &[f64],
    d: usize,
    (i0, i1, j0, j1): (usize, usize, usize, usize),
    upper: bool,
) -> f64 {
    // Above the diagonal nothing left of column i0 + 1 is needed, so the
    // product is not formed there either.
    let j0 = if upper { j0.max(i0 + 1) } else { j0 };
    let mut g = buf.slice_mut(s![..i1 - i0, ..j1 - j0]);
    general_mat_mul(1.0, &rows(x, d, i0, i1), &rows(y, d, j0, j1).t(), 0.0, &mut g);
    let mut s = 0.0;
    for (i, row) in (i0..i1).zip(g.rows()) {
        let first = if upper { (i + 1).max(j0) } else { j0 };
        let row = &row.as_slice().expect("tile rows are contiguous")[first - j0..];
        let ny = &ny[first..j1];
        let mut acc = [0.0; LANES];
        let mut dots = row.chunks_exact(LANES);
        let mut nys = ny.chunks_exact(LANES);
        for (dot, ny) in (&mut dots).zip(&mut nys) {
            for l in 0..LANES {
                acc[l] += (nx[i] + ny[l] - 2.0 * dot[l]).max(0.0).sqrt();
            }
        }
        for (dot, ny) in dots.remainder().iter().zip(nys.remainder()) {
            acc[0] += (nx[i] + ny - 2.0 * dot).max(0.0).sqrt();
        }
        s += acc.iter().sum::<f64>();
    }
    s
}

/// Mean Euclidean distance over all pairs (rows of `x` against rows of `y`).
fn mean_cross(x: &[f64], nx: &[f64], y: &[f64], ny: &[f64], d: usize) -> f64 {
    let total: f64 = tiles(nx.len(), ny.len(), false)
        .into_par_iter()
        .map_init(tile_buffer, |buf, t| tile_sum(buf, x, nx, y, ny, d, t, false))
        .sum();
    total / (nx.len() as f64 * ny.len() as f64)
}

/// Mean Euclidean distance over all distinct pairs within `x` (row-major
/// `n x d`). Runs on the current pool.
pub fn self_term(x: &[f64], d: usize) -> f64 {
    let nx = norms(x, d);
    mean_self(x, &nx, d)
}

/// Mean Euclidean distance over all distinct pairs within `x`. Each
/// unordered pair is visited once, and the diagonal never enters: its
/// expansion is a rounding residual, not zero, and its square root would
/// not be negligible.
fn mean_self(x: &[f64], nx: &[f64], d: usize) -> f64 {
    let n = nx.len();
    if n < 2 {
        return 0.0;
    }
    let total: f64 = tiles(n, n, true)
        .into_par_iter()
        .map_init(tile_buffer, |buf, t| tile_sum(buf, x, nx, x, nx, d, t, true))
        .sum();
    2.0 * total / (n as f64 * (n as f64 - 1.0))
}

/// `x` row-major `n x d`, `y` row-major `m x d`. The within-sample terms
/// are computed only where requested, so a fixed sample's term can be
/// reused across comparisons. Runs on the current pool.
pub fn energy(x: &[f64], y: &[f64], d: usize, need_x: bool, need_y: bool) -> Energy {
    let nx = norms(x, d);
    let ny = norms(y, d);
    // The terms are joined rather than run in sequence so their tiles form
    // one wave over the pool, with one idle tail instead of three.
    let (cross, (self_x, self_y)) = rayon::join(
        || mean_cross(x, &nx, y, &ny, d),
        || {
            rayon::join(
                || need_x.then(|| mean_self(x, &nx, d)),
                || need_y.then(|| mean_self(y, &ny, d)),
            )
        },
    );
    Energy { cross, self_x, self_y }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize, d: usize, seed: f64) -> Vec<f64> {
        (0..n * d).map(|i| (i as f64 * 0.37 + seed).sin()).collect()
    }

    /// The pairwise formulation, one distance at a time, for agreement.
    fn direct(x: &[f64], y: &[f64], d: usize, upper: bool) -> f64 {
        let mut s = 0.0;
        for (i, a) in x.chunks_exact(d).enumerate() {
            for (j, b) in y.chunks_exact(d).enumerate() {
                if !upper || j > i {
                    s += a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
                }
            }
        }
        s
    }

    #[test]
    fn identical_samples_have_zero_energy() {
        let x = cloud(20, 2, 0.0);
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
        let x = cloud(20, 3, 0.0);
        let y: Vec<f64> = x.iter().map(|v| v + 0.5).collect();
        let z: Vec<f64> = x.iter().map(|v| v + 2.0).collect();
        let ey = energy(&x, &y, 3, true, true);
        let ez = energy(&x, &z, 3, true, true);
        let ed = |e: &Energy| 2.0 * e.cross - e.self_x.unwrap() - e.self_y.unwrap();
        assert!(ed(&ez) > ed(&ey));
    }

    #[test]
    fn tiles_agree_with_the_pairwise_sums() {
        // Sizes chosen so both samples span several row tiles and the
        // second spans two column tiles, with ragged last tiles.
        let (n, m, d) = (150, COL_BLOCK + 37, 5);
        let x = cloud(n, d, 0.1);
        let y = cloud(m, d, 0.9);
        let e = energy(&x, &y, d, true, true);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        assert!(close(e.cross, direct(&x, &y, d, false) / (n * m) as f64));
        let pairs = |k: usize| (k * (k - 1)) as f64 / 2.0;
        assert!(close(e.self_x.unwrap(), direct(&x, &x, d, true) / pairs(n)));
        assert!(close(e.self_y.unwrap(), direct(&y, &y, d, true) / pairs(m)));
    }

    #[test]
    fn one_column_and_one_row() {
        let x = cloud(70, 1, 0.2);
        let y = cloud(3, 1, 0.5);
        let e = energy(&x, &y, 1, true, true);
        assert!((e.cross - direct(&x, &y, 1, false) / 210.0).abs() < 1e-12);
        assert_eq!(self_term(&y[..1], 1), 0.0);
        assert_eq!(energy(&x[..1], &y, 1, true, false).self_x, Some(0.0));
    }
}
