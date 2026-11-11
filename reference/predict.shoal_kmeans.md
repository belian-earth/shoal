# Assign New Observations to Fitted Clusters

Assigns each row of `newdata` to the nearest centroid by squared
Euclidean distance, or for a cosine
[`shoal_emeans()`](https://belian-earth.github.io/shoal/reference/shoal_emeans.md)
model to the centroid of largest cosine similarity. For a
[`shoal_emeans()`](https://belian-earth.github.io/shoal/reference/shoal_emeans.md)
model, predicting the training data on the machine that fitted it
reproduces its `cluster` exactly.

## Usage

``` r
# S3 method for class 'shoal_kmeans'
predict(object, newdata = NULL, ...)

# S3 method for class 'shoal_emeans'
predict(object, newdata = NULL, ...)
```

## Arguments

- object:

  A fitted
  [`shoal_kmeans()`](https://belian-earth.github.io/shoal/reference/shoal_kmeans.md)
  or
  [`shoal_emeans()`](https://belian-earth.github.io/shoal/reference/shoal_emeans.md)
  model.

- newdata:

  A numeric matrix or data frame with the same columns as the data the
  model was fitted to. Omit it to return the training assignments.

- ...:

  Ignored.

## Value

An integer vector of cluster IDs, one per row of `newdata`.

## Details

Only algorithms with a notion of a cluster centre can do this. The
density-based methods deliberately have no
[`predict()`](https://rdrr.io/r/stats/predict.html) method, so calling
[`predict()`](https://rdrr.io/r/stats/predict.html) on their results
raises R's standard "no applicable method" error rather than a bespoke
one.

## Examples

``` r
fit <- shoal_kmeans(as.matrix(iris[1:100, 1:4]), k = 2L)
predict(fit, as.matrix(iris[101:150, 1:4]))
#>  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#> [39] 2 2 2 2 2 2 2 2 2 2 2 2
```
