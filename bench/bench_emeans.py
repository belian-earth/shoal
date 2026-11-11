"""Benchmark: e-means and k-means in Python, then a score for every method.

Run from project root, after bench_emeans.R (which writes the data):
  uv run bench/bench_emeans.py

Settings match bench_emeans.R: one start from k random rows, at most 25
Lloyd iterations, and 5 passes at the e-means batch size for the mini-batch
methods. faiss and the reference e-means run in float32, as they are built
to; scikit-learn runs in float64, as shoal does. Every method uses all the
training rows: faiss's k-means would otherwise subsample to 256 per centroid.

Each method's centroids, from R and from here, are then scored by one
evaluator, in float64: the mean squared distance from each row to its
nearest centroid, on the training rows and on 50,000 held-out rows. The
baseline is a converged Lloyd run, 100 iterations of faiss's k-means.

faiss's OpenMP threads time erratically on a hybrid CPU with performance
and efficiency cores, and its default of every logical core was the slowest
setting here. It is therefore timed at 6, 12 and 20 threads and keeps the
fastest. shoal and scikit-learn use every core; their timings repeat to
within a few percent.

All of this is on CPU. The reference e-means is fastest on a GPU, through
PyTorch with Triton or CuTe kernels; shoal has no GPU path and is not
compared with one.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26",
#     "scikit-learn>=1.5",
#     "faiss-cpu==1.16.0",
# ]
# ///

import csv
import gc
import os
import re
import time
import warnings
from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from sklearn.cluster import KMeans, MiniBatchKMeans

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data" / "emeans"
D = 64
KS = (256, 1024)
LLOYD_ITERS = 25
EPOCHS = 5
SEED = 1
FAISS_THREADS = (6, 12, 20)


def load(name):
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"No {path}. Run bench/bench_emeans.R first.")
    return np.fromfile(path, dtype=np.float64).reshape(-1, D)


def time_fit(fn, reps=3, slow=30.0):
    """Median of three, or a single run when the first takes over `slow` s."""
    times = []
    while True:
        gc.collect()
        t0 = time.perf_counter()
        centroids = fn()
        times.append(time.perf_counter() - t0)
        if len(times) == reps or times[0] > slow:
            break
    times.sort()
    return times[len(times) // 2], len(times), np.asarray(centroids, dtype=np.float64)


def best_threads(fn):
    """time_fit() at each faiss thread count, keeping the fastest."""
    best = None
    for th in FAISS_THREADS:
        faiss.omp_set_num_threads(th)
        timed = time_fit(fn)
        if best is None or timed[0] < best[0][0]:
            best = (timed, th)
    faiss.omp_set_num_threads(max(FAISS_THREADS))
    return best


def faiss_lloyd(x32, k, niter):
    km = faiss.Kmeans(D, k, niter=niter, nredo=1, seed=SEED,
                      max_points_per_centroid=x32.shape[0] // k + 1)
    km.train(x32)
    return km.centroids


def mse(x, c, chunk=65536):
    """Mean squared distance from each row of x to its nearest row of c."""
    cc = np.einsum("ij,ij->i", c, c)
    total = 0.0
    for i in range(0, x.shape[0], chunk):
        b = x[i:i + chunk]
        d2 = np.einsum("ij,ij->i", b, b)[:, None] - 2.0 * (b @ c.T) + cc[None, :]
        total += np.maximum(d2.min(axis=1), 0.0).sum()
    return total / x.shape[0]


def main():
    x = load("train.f64")
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    test = load("test.f64")
    threads = os.cpu_count()

    rows = []

    def record(k, method, package, timed, threads=threads):
        seconds, runs, c = timed
        slug = re.sub(r"[^a-z0-9]+", "-", f"{method} {package}".lower()).strip("-")
        file = f"{slug}_k{k}_py.f64"
        np.ascontiguousarray(c).tofile(DATA / file)
        rows.append(dict(k=k, method=method, package=package, threads=threads,
                         seconds=seconds, runs=runs, centroids=file))
        print(f"  {method:28s} {package:14s} {seconds:8.2f} s ({runs} run{'' if runs == 1 else 's'})",
              flush=True)

    for k in KS:
        print(f"\n=== k = {k} ===", flush=True)

        ems = {}

        def reference():
            ems["em"] = NumpyEmeans(D, k, nepochs=EPOCHS, seed=SEED)
            return ems["em"].train(x32)[0]

        timed, th = best_threads(reference)
        record(k, "e-means", "faiss (reference)", timed, th)
        batch = ems["em"].recipe["batch_size"]

        timed, th = best_threads(lambda: faiss_lloyd(x32, k, LLOYD_ITERS))
        record(k, "Lloyd", "faiss", timed, th)

        record(k, "Lloyd", "scikit-learn", time_fit(lambda: KMeans(
            n_clusters=k, init="random", n_init=1, max_iter=LLOYD_ITERS,
            algorithm="lloyd", random_state=SEED).fit(x).cluster_centers_))

        # The e-means budget: its batch size for 5 passes, early stopping off.
        record(k, "mini-batch", "scikit-learn", time_fit(lambda: MiniBatchKMeans(
            n_clusters=k, init="random", n_init=1, batch_size=batch,
            max_iter=EPOCHS, max_no_improvement=None, tol=0.0,
            random_state=SEED).fit(x).cluster_centers_))

        # The loss baseline; its time is incidental.
        faiss.omp_set_num_threads(12)
        record(k, "Lloyd, converged", "faiss",
               time_fit(lambda: faiss_lloyd(x32, k, 100), reps=1), 12)
        faiss.omp_set_num_threads(max(FAISS_THREADS))

    # Score every set of centroids, from R and from here, the same way.
    r_results = HERE / "results_emeans_r.csv"
    if r_results.exists():
        with r_results.open(newline="") as f:
            rows = [dict(r, k=int(r["k"])) for r in csv.DictReader(f)] + rows
    else:
        print(f"\nNo {r_results}; scoring the Python runs only.")

    for r in rows:
        c = np.fromfile(DATA / r["centroids"], dtype=np.float64).reshape(-1, D)
        r["train_mse"] = mse(x, c)
        r["test_mse"] = mse(test, c)
    for r in rows:
        base = next(b["test_mse"] for b in rows
                    if b["k"] == r["k"] and b["method"] == "Lloyd, converged")
        r["vs_converged_pct"] = 100.0 * (r["test_mse"] / base - 1.0)

    out = HERE / "results_emeans.csv"
    fields = ["k", "method", "package", "threads", "seconds", "runs",
              "train_mse", "test_mse", "vs_converged_pct"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["k"], float(r["seconds"]))))
    print(f"\nSaved: {out}\n")

    print("| k | Method | Package | Threads | Seconds | Held-out MSE vs converged Lloyd |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["k"], float(r["seconds"]))):
        print(f"| {r['k']} | {r['method']} | {r['package']} | {r['threads']} | "
              f"{float(r['seconds']):.1f} | {r['vs_converged_pct']:+.2f}% |")


if __name__ == "__main__":
    main()
