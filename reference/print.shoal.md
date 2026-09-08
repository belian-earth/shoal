# Print a clustering result

A single method serves every algorithm: the heading, parameters and
cluster counts all come from components the shared constructor
guarantees. Algorithms with extra output specialise and call
[`NextMethod()`](https://rdrr.io/r/base/UseMethod.html).

## Usage

``` r
# S3 method for class 'shoal_clustering'
print(x, ...)

# S3 method for class 'shoal_hdbscan'
print(x, ...)

# S3 method for class 'shoal_evoc'
print(x, ...)

# S3 method for class 'shoal_kmeans'
print(x, ...)

# S3 method for class 'shoal_gmm'
print(x, ...)

# S3 method for class 'shoal_slic'
print(x, ...)
```

## Arguments

- x:

  A clustering result.

- ...:

  Ignored.

## Value

`x`, invisibly.
