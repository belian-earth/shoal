# Benchmarks

Every clustering algorithm in shoal, the distance matrix and the
nearest-neighbour search are benchmarked against the best R alternative
and, where one exists, the Python one, at matched settings: the same k,
restarts, iteration cap, linkage and neighbourhood size. Timings are medians
of three runs on 20 cores and 62 GB, from 500 to 50,000 points, on Gaussian
blobs in 2 and 10 dimensions and, for EVoC, on embedding-like data in 48.

![Scaling benchmark](scaling.png)

## What the panels show

At 50,000 points unless noted.

- **DBSCAN**: 6 to 9 times faster than the dbscan package and 1.5 to 5 times
  faster than scikit-learn, the gap growing with dimension.
- **HDBSCAN**: 20 times faster than scikit-learn in 2 dimensions and 2 times
  in 10. The dbscan package builds a full distance matrix and is 40 times
  slower at 20,000 points, its largest feasible size here.
- **k-means**: level with base R's Hartigan-Wong in 2 dimensions and 2 times
  faster in 10, with identical within-cluster sums of squares. scikit-learn's
  times are erratic at these sizes because its OpenMP threading oversubscribes
  a 20-core machine.
- **Gaussian mixture**: 20 to 170 times faster than mclust, which fits a
  hierarchical initialisation first, and level with scikit-learn from about
  5,000 points, having been several times faster below that.
- **Ward**: 2.5 to 4.5 times faster than base R and about 2 times faster
  than SciPy. All three run the same nearest-neighbour-chain algorithm, so the
  differences are in computing the distances that feed it. Given raw data,
  `shoal_hclust()` writes the distances straight into the buffer the
  clustering consumes, so at 50,000 points it holds one 10 GB vector where
  `stats::dist()` plus `stats::hclust()` holds two, and that saved copy is
  most of the gap.
- **Distances**: `shoal_dist()` is 2 times faster than `stats::dist()` in 2
  dimensions and 4.5 times in 10, and level with SciPy's `pdist()`, which
  edges it in 2 dimensions where the work is almost all memory traffic. The
  result is written straight into the R vector, one pass, in parallel.
- **kNN**: `shoal_knn()` is 10 times faster than both `dbscan::kNN()` and
  scikit-learn's `NearestNeighbors` in 2 dimensions, where all three use a
  kd-tree, and 2.5 to 3 times faster in 10, where shoal's `search = "auto"`
  has switched to a parallel scan and the others are still in a tree that
  prunes little. Both paths return the same result, ties included, and
  agree with `shoal_dist()` to the bit.
- **EVoC**: level with the reference implementation from about 20,000 points,
  and several times faster below that, where the reference's compiled kernels
  have not amortised. The reference here runs on 20 numba threads.

## Wide data

- The spatial indexes behind DBSCAN, HDBSCAN and `shoal_knn()` degrade as
  dimension grows, as they do everywhere. `shoal_knn()` switches from its
  kd-tree to a parallel scan above 8 columns, where the scan is several times
  faster than a tree. Above a few dozen columns, HDBSCAN's default Boruvka
  tree search is slower than the plain alternative, so pass `boruvka = FALSE`
  there. See `?shoal_hdbscan`.
- For high-dimensional embedding vectors, `shoal_evoc()` is the right tool
  and is orders of magnitude faster than HDBSCAN on the raw vectors.

## e-means at scale

The scaling benchmark above stops at five clusters and 50,000 rows, below
the scale e-means is for. `bench_emeans.R` and `bench_emeans.py` cluster
500,000 rows of embedding-like data, unit vectors in 64 dimensions around
2,000 topic directions, into 256 and 1024 clusters. Speed alone would favour
a method that stops early, so one evaluator scores every method's centroids
on 50,000 held-out rows: the mean squared distance to the nearest centroid,
relative to a converged Lloyd run of 100 iterations in faiss. A negative
value is a lower loss than the converged run.

Every method starts once from k random rows. Lloyd-type methods run at most
25 iterations, the faiss default. The mini-batch methods run five passes at
the batch size e-means chose, and e-means runs its default five passes.

| Method | Package | Threads | k = 256 | Loss | k = 1024 | Loss |
|----|----|--:|--:|--:|--:|--:|
| e-means | shoal | 20 | 0.8 s | -0.86% | 2.2 s | -1.60% |
| e-means | faiss (reference) | 12 | 2.0 s | -0.22% | 3.6 s | -1.48% |
| mini-batch | scikit-learn | 20 | 1.8 s | +2.04% | 4.4 s | +1.35% |
| Lloyd | scikit-learn | 20 | 2.9 s | +0.13% | 10.5 s | +1.24% |
| Lloyd | faiss | 12 | 3.1 s | +0.42% | 11.1 s | +0.01% |
| Lloyd | shoal | 20 | 8.9 s | +0.71% | 38.7 s | +1.75% |
| Hartigan-Wong | stats | 1 | 17.2 s | +3.82% | 126.3 s | +1.73% |
| mini-batch | ClusterR | 1 | 37.5 s | -0.11% | 139.6 s | +2.69% |

- `shoal_emeans()` is the fastest method at both k, and the two e-means
  implementations reach the lowest losses, below converged Lloyd. It is 1.6
  to 2.6 times faster than the reference implementation on the same CPU,
  and 3.7 to 4.8 times faster than the fastest Lloyd run, which it beats on
  loss by 1 to 3 percentage points. At matched batch size and passes it is
  2 to 2.3 times faster than scikit-learn's mini-batch k-means, at a loss
  about 3 points lower.
- Over five seeds each, the two e-means implementations are level at
  k = 1024, with mean losses of -1.43% for shoal and -1.61% for the
  reference. At k = 256 shoal's mean is 0.4 points lower, -1.06% against
  -0.62%, a gap similar to the spread of the reference's seeds. Over 80
  seeds on AlphaEarth embeddings their mean losses agreed to 0.01%.
- `shoal_kmeans()` defaults to a k-means++ start, which here costs more than
  the iterations: 57 s in all at k = 256, against 8.9 s from random rows,
  for a loss 0.9 points lower. Its cost grows with the square of k, because
  linfa recomputes each row's distance to every centre chosen so far, and at
  k = 1024 it had not finished after nine minutes. The table therefore uses
  random starts throughout.
- Base R's Hartigan-Wong stops after one to seven iterations at this size,
  warning that its quick-transfer stage exceeded its step limit, so its row
  measures an aborted fit. ClusterR's mini-batch k-means is single-threaded.
- faiss's OpenMP threads time erratically on this hybrid CPU of 6
  performance and 8 efficiency cores, and its default of all 20 was its
  slowest setting. It is timed at 6, 12 and 20 threads and keeps its
  fastest. faiss and the reference run in float32, the rest in float64.
  shoal's time includes a final exact assignment pass that the reference
  skips.
- One dataset, and one seed per method in the table.

**These are CPU timings only.** The e-means paper's fastest results are on
GPUs, where the reference runs through PyTorch with Triton or CuTe kernels,
and faiss also has a GPU k-means. shoal has no GPU path and has not been
compared with either, so nothing here is a claim about GPU performance.

## Running the benchmarks

Everything runs from the project root through the Makefile in this
directory:

```sh
make -C bench            # data, release install, R and Python runs, collation
make -C bench bench-r    # the R side only
make -C bench compare    # tables and the figure from existing results
make -C bench bench-emeans  # the e-means benchmark, about 15 minutes
```

The steps behind those targets:

```sh
Rscript bench/gen_data.R                  # shared datasets, once
NOT_CRAN=true R CMD INSTALL .             # release build; see the note below
Rscript bench/bench_r.R                   # shoal against the R alternatives
uv run bench/bench_sklearn.py             # the Python alternatives
Rscript bench/compare.R                   # speedup tables and scaling.png
```

`BENCH_ONLY=<algorithm>` on either benchmark script reruns one algorithm and
merges its rows into the existing results, so a change to one function does
not mean rerunning everything. The names are those in the figure: `DBSCAN`,
`HDBSCAN`, `k-means`, `GMM`, `Ward`, `Distances`, `kNN`, `EVoC`.

The results files are not committed; only the figure is. The 50,000-point
Ward and Distances rows hold 10 GB distance vectors, two at a time for Ward,
so budget 30 GB of RAM for a full run. R alternatives run in a subprocess so
that one crashing costs a row rather than the run, and any alternative known
to fail at scale is capped and reported as missing above its cap.

Timings from a `devtools::load_all()` session are meaningless: it compiles
the Rust code without optimisation, and everything runs an order of
magnitude slower or worse. Install the package first, as above.

## Adding a benchmark

Three edits, one per script:

1. An entry in the `benchmarks` list in `bench_r.R`: the shoal call, the
   alternative, its package name and the largest size to run it at.
2. An entry in `BENCHMARKS` in `bench_sklearn.py` with matched settings.
3. The algorithm's name in the `algorithms` vector in `compare.R`, which
   fixes the panel order, and a style entry there if the alternative's
   package is new.

Then `BENCH_ONLY=<name>` on both scripts and `Rscript bench/compare.R`.
