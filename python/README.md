# SuperKMeans Python Bindings

Python bindings for SuperKMeans

### Install from source

```bash
git clone https://github.com/cwida/SuperKMeans.git
cd SuperKMeans
pip install .
```

## Quick Start

```python
import numpy as np
from superkmeans import SuperKMeans

n = 100000
d = 512
k = 1000

data = np.random.randn(n, d).astype(np.float32)

kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d
)

# Compute centroids
centroids = kmeans.train(data)

# Assign points to clusters
assignments = kmeans.assign(data, centroids)
```

### Quantized clustering

Cluster in a compressed domain (`sq8`, `lvq4`, or `rabitq`) for faster clustering with a lower memory footprint:

```python
kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d,
    quantizer="sq8"
)

centroids = kmeans.train(data)

# Assign in the compressed domain
assignments = kmeans.quantized_assign(data, centroids)
```

## API Reference

### SuperKMeans Class

#### Constructor Parameters

- `n_clusters` (int): Number of clusters to create
- `dimensionality` (int): Number of dimensions in the data
- `quantizer` (str, default="f32"): Quantization scheme: "f32", "sq8", "lvq4", or "rabitq"
- `hierarchical` (bool, default=None): Use hierarchical clustering. If None, auto-enabled for n > 100,000
- `iters` (int, default=10): Number of k-means iterations (flat mode)
- `iters_mesoclustering` (int, default=3): Number of mesoclustering iterations (hierarchical mode)
- `iters_fineclustering` (int, default=5): Number of fineclustering iterations (hierarchical mode)
- `iters_refinement` (int, default=0): Number of refinement iterations (hierarchical mode)
- `sampling_fraction` (float, default=None): Fraction of data to sample (0.0, 1.0]. If None, uses 0.3 (flat) or 1.0 (hierarchical)
- `max_points_per_cluster` (int, default=256): Maximum points per cluster to sample
- `n_threads` (int, default=0): Number of threads (0 = use all)
- `seed` (int, default=42): Random seed for reproducibility
- `use_blas_only` (bool, default=False): Disable pruning, use BLAS only
- `full_precision_final_centroids` (bool, default=False): Recompute final centroids in full precision (quantized only)
- `tol` (float, default=1e-4): Tolerance for shift-based early stopping
- `recall_tol` (float, default=0.005): Tolerance for recall-based early stopping
- `early_termination` (bool, default=True): Enable early stopping
- `sample_queries` (bool, default=False): Sample queries from data
- `objective_k` (int, default=100): Number of neighbors for recall
- `verbose` (bool, default=False): Print progress information
- `angular` (bool, default=False): Use spherical k-means

#### Methods

**`train(data, queries=None)`**

Run k-means clustering to compute centroids.

- **Parameters:**
  - `data` (ndarray): Shape (n_samples, dimensionality), dtype float32
  - `queries` (ndarray, optional): Shape (n_queries, dimensionality), dtype float32
- **Returns:** `centroids` (ndarray): Shape (n_clusters, dimensionality), dtype float32

**`assign(vectors, centroids)`**

Assign vectors to nearest centroids in full precision. Can be called before `train()`. Aliased as `add()`.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

**`quantized_assign(vectors, centroids)`**

Assign vectors in the quantizer's compressed domain. Standalone — fits a fresh quantizer on the input vectors (reusing no trained state), so it can be called before `train()`. Equivalent to `assign()` when `quantizer="f32"`.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

**`assign_training_points(vectors, centroids)`**

Fast assignment of the training data, reusing state from `train()`. `vectors` must be the same data passed to `train()`.

- **Parameters:**
  - `vectors` (ndarray): Shape (n_vectors, dimensionality), dtype float32
  - `centroids` (ndarray): Shape (n_centroids, dimensionality), dtype float32
- **Returns:** `assignments` (ndarray): Shape (n_vectors,), dtype uint32

#### Properties

- `n_clusters_` (int): Number of clusters (read-only)
- `is_trained_` (bool): Whether model has been trained (read-only)
- `hierarchical_` (bool): Whether hierarchical mode is used (read-only)
- `quantizer_` (str): Quantization scheme in use (read-only)
- `iteration_stats` (list): List of `SuperKMeansIterationStats` objects
- `hierarchical_iteration_stats` (`HierarchicalSuperKMeansIterationStats`): Per-phase statistics (hierarchical mode only)

### SuperKMeansIterationStats Class

Statistics for a single iteration.

**Attributes:**
- `iteration` (int): Iteration number (1-indexed)
- `objective` (float): Within-cluster sum of squares (WCSS)
- `shift` (float): Average squared centroid shift
- `split` (int): Number of clusters split (empty cluster handling)
- `recall` (float): Recall@k value (0.0 to 1.0)
- `not_pruned_pct` (float): Percentage of vectors not pruned
- `partial_d` (int): Dimensions used for partial distance (d')
- `is_gemm_only` (bool): Whether iteration used BLAS-only

### HierarchicalSuperKMeansIterationStats Class

Per-phase statistics for hierarchical clustering.

**Attributes:**
- `mesoclustering_iteration_stats` (list): `SuperKMeansIterationStats` for the mesoclustering phase
- `fineclustering_iteration_stats` (list): `SuperKMeansIterationStats` for the fineclustering phase
- `refinement_iteration_stats` (list): `SuperKMeansIterationStats` for the refinement phase

## Examples

See the `examples/` directory for complete examples:
- `simple_clustering.py [n] [d] [k]`
- `quantized_clustering.py [quantizer] [n] [d] [k]`

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest python/tests/ -v

```
