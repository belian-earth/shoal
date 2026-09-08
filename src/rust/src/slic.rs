//! Graph SLIC: superpixel relaxation on a point cloud with a neighbour graph.
//!
//! SLIC (Achanta et al. 2012) grows compact, feature-homogeneous patches
//! from a regular seeding by repeatedly moving boundary points to the
//! adjacent patch whose feature mean is closest, with a spatial term that
//! keeps patches compact. This is the same relaxation on an arbitrary
//! point set: the raster's pixel adjacency is replaced by a caller-supplied
//! neighbour matrix and its grid seeding by caller-supplied initial labels.
//! Nothing here knows about rasters or grids, so it serves a hexagonal or
//! pentagonal discrete global grid as well as a raster.
//!
//! The objective for moving point `i` to patch `p` is
//!
//! `||x_i - m_p||^2 + lambda_p * ||xy_i - c_p||^2 / s_nom^2`
//!
//! with `m_p` and `c_p` the feature and coordinate means of the patch at
//! the start of the iteration (all points move together, Jacobi style).
//! `lambda_p` is one global value, or, in the adaptive mode, `alpha` times
//! the patch's own mean squared feature distance (ASLIC-style: the balance
//! between the two terms is then relative to local contrast).
//!
//! After the iterations every patch is split into its connected components
//! and patches below `min_patch` points are absorbed into the neighbouring
//! patch they share the most edges with. That pass runs sequentially in
//! ascending patch id over labels read before the pass, so a small patch
//! absorbed into a later small patch travels with it; those are the
//! semantics of the R implementation this replaces, kept so the two agree
//! exactly.
//!
//! Every step is deterministic: ties go to the point's own patch, then to
//! the earlier neighbour slot, and component ids follow first appearance in
//! point order.

use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Adaptive {
    None,
    Mean,
}

impl Adaptive {
    /// Names are validated on the R side; an unknown name here is a bug.
    pub fn from_name(name: &str) -> Self {
        match name {
            "none" => Adaptive::None,
            "mean" => Adaptive::Mean,
            _ => panic!("Unknown adaptive mode: {name}"),
        }
    }
}

pub struct Params {
    pub lambda: f64,
    pub s_nom: f64,
    pub n_iter: usize,
    pub adaptive: Adaptive,
    pub alpha: f64,
    pub min_patch: usize,
    /// Stop early once fewer than this share of points move in an iteration.
    pub tol: f64,
}

pub struct Output {
    /// One-based patch labels, compacted to `1..=n_patches`.
    pub labels: Vec<i32>,
    /// Points that changed patch in each iteration run.
    pub moves: Vec<i32>,
    /// Mean squared feature distance of each initial patch's points to its
    /// mean, indexed by initial label: the local-variability statistic a
    /// data-driven `lambda` is scaled from.
    pub init_stat: Vec<f64>,
    /// Median of `lambda_p` over patches in each iteration run (constant in
    /// the fixed mode).
    pub lambda_median: Vec<f64>,
    pub n_patches: usize,
}

/// Neighbour matrix, row-major `n x k`, zero-based, `-1` for a missing slot.
pub struct Graph<'a> {
    pub nb: &'a [i32],
    pub k: usize,
}

impl Graph<'_> {
    #[inline(always)]
    fn row(&self, i: usize) -> &[i32] {
        &self.nb[i * self.k..(i + 1) * self.k]
    }
}

/// `x` row-major `n x d`, `xy` row-major `n x 2`, `init` zero-based dense
/// labels. Runs on the current rayon pool.
pub fn graph_slic(x: &[f64], d: usize, xy: &[f64], g: &Graph, init: &[i32], p: &Params) -> Output {
    let n = init.len();
    assert_eq!(x.len(), n * d);
    assert_eq!(xy.len(), n * 2);
    assert_eq!(g.nb.len(), n * g.k);
    let s2 = p.s_nom * p.s_nom;

    let mut labels: Vec<i32> = init.to_vec();
    let mut next = vec![0i32; n];
    let mut moves = Vec::with_capacity(p.n_iter);
    let mut lambda_median = Vec::with_capacity(p.n_iter);
    let mut init_stat = Vec::new();

    for it in 0..p.n_iter {
        let np = n_patches(&labels);
        let (pm, pc, cnt) = patch_means(x, d, xy, &labels, np);

        // Per-patch lambda. The mean squared distance to the patch mean is
        // the init statistic at iteration 0 and the adaptive weight always.
        let need_stat = it == 0 || p.adaptive == Adaptive::Mean;
        let stat = if need_stat {
            let d2: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| sq_dist(&x[i * d..(i + 1) * d], &pm[labels[i] as usize * d..(labels[i] as usize + 1) * d]))
                .collect();
            let mut s = vec![0.0; np];
            for i in 0..n {
                s[labels[i] as usize] += d2[i];
            }
            for l in 0..np {
                if cnt[l] > 0 {
                    s[l] /= cnt[l] as f64;
                }
            }
            s
        } else {
            Vec::new()
        };
        if it == 0 {
            init_stat = stat.clone();
        }
        let lam: Vec<f64> = match p.adaptive {
            Adaptive::None => vec![p.lambda; np],
            Adaptive::Mean => stat.iter().map(|&s| p.alpha * s.max(1e-4)).collect(),
        };
        lambda_median.push(median(&lam));

        // One independent decision per boundary point.
        let dist = |i: usize, l: usize| -> f64 {
            let f = sq_dist(&x[i * d..(i + 1) * d], &pm[l * d..(l + 1) * d]);
            let dx = xy[i * 2] - pc[l * 2];
            let dy = xy[i * 2 + 1] - pc[l * 2 + 1];
            f + lam[l] * (dx * dx + dy * dy) / s2
        };
        let labels_ref = &labels;
        next.par_iter_mut().enumerate().for_each(|(i, out)| {
            let p0 = labels_ref[i];
            let nbi = g.row(i);
            let boundary = nbi.iter().any(|&j| j >= 0 && labels_ref[j as usize] != p0);
            if !boundary {
                *out = p0;
                return;
            }
            let mut best = p0;
            let mut bd = dist(i, p0 as usize);
            for &j in nbi {
                if j < 0 {
                    continue;
                }
                let pj = labels_ref[j as usize];
                if pj == p0 {
                    continue;
                }
                let dj = dist(i, pj as usize);
                if dj < bd {
                    bd = dj;
                    best = pj;
                }
            }
            *out = best;
        });
        let moved = labels.iter().zip(next.iter()).filter(|(a, b)| a != b).count();
        moves.push(moved as i32);
        std::mem::swap(&mut labels, &mut next);
        if (moved as f64) < p.tol * n as f64 {
            break;
        }
    }

    split_components(&mut labels, g);
    absorb_dust(&mut labels, g, p.min_patch);
    let n_patches = compact(&mut labels);

    Output {
        labels: labels.iter().map(|&l| l + 1).collect(),
        moves,
        init_stat,
        lambda_median,
        n_patches,
    }
}

#[inline(always)]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for c in 0..a.len() {
        let t = a[c] - b[c];
        s += t * t;
    }
    s
}

fn n_patches(labels: &[i32]) -> usize {
    labels.iter().copied().max().map_or(0, |m| m as usize + 1)
}

/// Feature and coordinate means per patch. Sequential: the reduction is a
/// small fraction of the per-point work and needs no per-thread copies of
/// an `np x d` accumulator.
fn patch_means(x: &[f64], d: usize, xy: &[f64], labels: &[i32], np: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let n = labels.len();
    let mut pm = vec![0.0; np * d];
    let mut pc = vec![0.0; np * 2];
    let mut cnt = vec![0usize; np];
    for i in 0..n {
        let l = labels[i] as usize;
        cnt[l] += 1;
        let row = &x[i * d..(i + 1) * d];
        let acc = &mut pm[l * d..(l + 1) * d];
        for c in 0..d {
            acc[c] += row[c];
        }
        pc[l * 2] += xy[i * 2];
        pc[l * 2 + 1] += xy[i * 2 + 1];
    }
    for l in 0..np {
        if cnt[l] > 0 {
            let k = cnt[l] as f64;
            for v in &mut pm[l * d..(l + 1) * d] {
                *v /= k;
            }
            pc[l * 2] /= k;
            pc[l * 2 + 1] /= k;
        }
    }
    (pm, pc, cnt)
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// Union-find with path halving.
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let gp = parent[parent[i as usize] as usize];
        parent[i as usize] = gp;
        i = gp;
    }
    i
}

/// Relabel each patch's connected components as separate patches. Component
/// ids follow the first point of each component in point order, which is
/// how igraph numbers them.
fn split_components(labels: &mut [i32], g: &Graph) {
    let n = labels.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        for &j in g.row(i) {
            if j >= 0 && labels[j as usize] == labels[i] {
                let a = find(&mut parent, i as u32);
                let b = find(&mut parent, j as u32);
                if a != b {
                    // Root the later one under the earlier: the root id is
                    // then irrelevant to numbering, which is by first point.
                    if a < b {
                        parent[b as usize] = a;
                    } else {
                        parent[a as usize] = b;
                    }
                }
            }
        }
    }
    let mut id_of_root = vec![-1i32; n];
    let mut next = 0i32;
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if id_of_root[r] < 0 {
            id_of_root[r] = next;
            next += 1;
        }
        labels[i] = id_of_root[r];
    }
}

/// Absorb patches smaller than `min_patch` into the neighbouring patch
/// they share the most edges with (ties: lowest id). Sequential in
/// ascending patch id over labels read before the pass; cells relabelled
/// to a later dust patch move again with it. Repeats until nothing moves.
fn absorb_dust(labels: &mut [i32], g: &Graph, min_patch: usize) {
    let n = labels.len();
    loop {
        let np = n_patches(labels);
        let mut cnt = vec![0usize; np];
        for &l in labels.iter() {
            cnt[l as usize] += 1;
        }
        let dust: Vec<usize> = (0..np).filter(|&l| cnt[l] > 0 && cnt[l] < min_patch).collect();
        if dust.is_empty() {
            break;
        }
        let mut is_dust = vec![false; np];
        for &l in &dust {
            is_dust[l] = true;
        }
        let stale: Vec<i32> = labels.to_vec();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); np];
        for i in 0..n {
            let l = stale[i] as usize;
            if is_dust[l] {
                groups[l].push(i);
            }
        }
        let mut counter = vec![0u32; np];
        let mut moved = false;
        for &p in &dust {
            let cells = std::mem::take(&mut groups[p]);
            if cells.is_empty() {
                continue;
            }
            let mut touched: Vec<usize> = Vec::new();
            for &i in &cells {
                for &j in g.row(i) {
                    if j < 0 {
                        continue;
                    }
                    let q = stale[j as usize] as usize;
                    if q == p {
                        continue;
                    }
                    if counter[q] == 0 {
                        touched.push(q);
                    }
                    counter[q] += 1;
                }
            }
            if touched.is_empty() {
                continue;
            }
            let mut best = usize::MAX;
            let mut bc = 0u32;
            for &q in &touched {
                if counter[q] > bc || (counter[q] == bc && q < best) {
                    bc = counter[q];
                    best = q;
                }
            }
            for &q in &touched {
                counter[q] = 0;
            }
            for &i in &cells {
                labels[i] = best as i32;
            }
            moved = true;
            if is_dust[best] && best > p {
                groups[best].extend(cells);
            }
        }
        if !moved {
            break;
        }
    }
}

/// Renumber labels to `0..n_patches` in ascending order of the old ids.
fn compact(labels: &mut [i32]) -> usize {
    let np = n_patches(labels);
    let mut seen = vec![false; np];
    for &l in labels.iter() {
        seen[l as usize] = true;
    }
    let mut map = vec![-1i32; np];
    let mut next = 0i32;
    for l in 0..np {
        if seen[l] {
            map[l] = next;
            next += 1;
        }
    }
    for l in labels.iter_mut() {
        *l = map[*l as usize];
    }
    next as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 6 x 6 grid of points with 4-neighbour adjacency, features 2-d:
    /// left half near (0, 0), right half near (1, 1). Seeded as 2 x 2 tiles.
    fn grid() -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>) {
        let side = 6usize;
        let n = side * side;
        let mut x = Vec::with_capacity(n * 2);
        let mut xy = Vec::with_capacity(n * 2);
        let mut nb = Vec::with_capacity(n * 4);
        let mut init = Vec::with_capacity(n);
        for r in 0..side {
            for c in 0..side {
                let v = if c < side / 2 { 0.0 } else { 1.0 };
                x.push(v + 0.01 * (r as f64));
                x.push(v);
                xy.push(c as f64 * 10.0);
                xy.push(r as f64 * 10.0);
                let at = |rr: isize, cc: isize| -> i32 {
                    if rr < 0 || cc < 0 || rr >= side as isize || cc >= side as isize {
                        -1
                    } else {
                        (rr as usize * side + cc as usize) as i32
                    }
                };
                nb.push(at(r as isize - 1, c as isize));
                nb.push(at(r as isize + 1, c as isize));
                nb.push(at(r as isize, c as isize - 1));
                nb.push(at(r as isize, c as isize + 1));
                init.push(((r / 2) * (side / 2) + c / 2) as i32);
            }
        }
        (x, xy, nb, init)
    }

    #[test]
    fn patches_follow_the_feature_seam() {
        let (x, xy, nb, init) = grid();
        let g = Graph { nb: &nb, k: 4 };
        let p = Params { lambda: 0.01, s_nom: 10.0, n_iter: 10, adaptive: Adaptive::None, alpha: 0.0, min_patch: 1, tol: 0.0 };
        let out = graph_slic(&x, 2, &xy, &g, &init, &p);
        // No patch straddles the seam between columns 2 and 3.
        for r in 0..6 {
            assert_ne!(out.labels[r * 6 + 2], out.labels[r * 6 + 3]);
        }
        assert_eq!(out.moves.len(), 10);
        assert_eq!(out.init_stat.len(), 9);
        assert!(out.labels.iter().all(|&l| l >= 1 && l as usize <= out.n_patches));
    }

    #[test]
    fn dust_is_absorbed_and_labels_are_dense() {
        let (x, xy, nb, init) = grid();
        let g = Graph { nb: &nb, k: 4 };
        let p = Params { lambda: 0.01, s_nom: 10.0, n_iter: 5, adaptive: Adaptive::Mean, alpha: 0.1, min_patch: 6, tol: 0.0 };
        let out = graph_slic(&x, 2, &xy, &g, &init, &p);
        let mut cnt = vec![0usize; out.n_patches];
        for &l in &out.labels {
            cnt[l as usize - 1] += 1;
        }
        assert!(cnt.iter().all(|&c| c >= 6), "{cnt:?}");
        assert_eq!(out.lambda_median.len(), 5);
    }

    #[test]
    fn components_are_numbered_by_first_point() {
        let mut labels = vec![0, 0, 1, 0, 0, 1];
        // chain 0-1-2-3-4-5
        let nb: Vec<i32> = vec![-1, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, -1];
        let g = Graph { nb: &nb, k: 2 };
        split_components(&mut labels, &g);
        assert_eq!(labels, vec![0, 0, 1, 2, 2, 3]);
    }
}
