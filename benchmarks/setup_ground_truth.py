"""
Generate ground truth for benchmark datasets.

This script computes the exact k-nearest neighbors for query vectors
using brute force search, and saves the results to JSON files.
"""

import json
import numpy as np
import os
import faiss
from sklearn import preprocessing
from bench_utils import (
    DATASET_PARAMS,
    ANGULAR_DATASETS,
    KNN_VALUES,
    N_QUERIES,
    get_data_path,
    get_query_path,
    get_ground_truth_path,
    GROUND_TRUTH_DIR
)


class BruteForceFAISS:
    """Brute force nearest neighbor search using FAISS."""

    def __init__(self, metric, dimension):
        if metric not in ("angular", "euclidean", "hamming", "ip"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)

        self.metric = metric
        self.dimension = dimension
        self.name = "BruteForceFAISS()"
        self.index = None

        # Set number of threads (you can adjust this)
        faiss.omp_set_num_threads(-1)  # Use all available threads

    def fit(self, X):
        """Build the index from training data.

        Args:
            X: Training data (n_samples, dimension) as float32 numpy array
        """
        if self.metric == "euclidean":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # Default to L2
            self.index = faiss.IndexFlatL2(self.dimension)

        # Ensure data is float32 and contiguous
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        self.index.add(X)

    def query_batch(self, queries, k):
        """Query the index with a batch of query vectors.

        Args:
            queries: Query vectors (n_queries, dimension) as float32 numpy array
            k: Number of nearest neighbors to return

        Returns:
            distances: (n_queries, k) array of distances
            indices: (n_queries, k) array of neighbor indices
        """
        if self.index is None:
            raise ValueError("Index not fitted. Call fit() first.")

        # Ensure data is float32 and contiguous
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)

        # FAISS search returns (distances, indices)
        distances, indices = self.index.search(queries, k)

        return distances, indices


def generate_ground_truth(dataset, knns=None, normalize=None):
    """Generate ground truth nearest neighbors for a dataset.

    Args:
        dataset: Dataset name (must be in DATASET_PARAMS)
        knns: List of k values for k-NN (default: KNN_VALUES from bench_utils)
        normalize: Whether to L2 normalize the data (default: True for angular datasets)
    """
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )

    # Use defaults if not specified
    if knns is None:
        knns = KNN_VALUES
    if normalize is None:
        normalize = dataset in ANGULAR_DATASETS

    # Create ground truth directory if it doesn't exist
    if not os.path.exists(GROUND_TRUTH_DIR):
        os.makedirs(GROUND_TRUTH_DIR)

    # Get dataset parameters
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]

    print(f"\n=== Generating ground truth for {dataset} ===")
    print(f"Dataset: {dataset}")
    print(f"Train vectors: {num_vectors:,}, dimensions: {num_dimensions}")
    print(f"Normalize: {normalize}")
    print(f"KNN values: {knns}")

    # Load train data
    train_path = get_data_path(dataset)
    print(f"\nLoading train data from: {train_path}")
    train = np.fromfile(train_path, dtype=np.float32)
    if train.size != num_vectors * num_dimensions:
        raise ValueError(
            f"Train file size mismatch: got {train.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    train = train.reshape(num_vectors, num_dimensions)
    print(f"Loaded train data: {train.shape}")

    # Load test/query data
    query_path = get_query_path(dataset)
    print(f"Loading query data from: {query_path}")
    test = np.fromfile(query_path, dtype=np.float32)

    # Infer number of queries from file size
    if test.size % num_dimensions != 0:
        raise ValueError(
            f"Query file size mismatch: {test.size} is not divisible by {num_dimensions}"
        )
    num_queries = test.size // num_dimensions
    test = test.reshape(num_queries, num_dimensions)
    test = test[:N_QUERIES, :]
    num_queries = len(test)
    print(f"Loaded query data: {test.shape}")
    print(f"Number of queries: {num_queries}")

    # Normalize if requested
    if normalize:
        print("\nNormalizing data (L2 norm)...")
        train = preprocessing.normalize(train, axis=1, norm='l2')
        test = preprocessing.normalize(test, axis=1, norm='l2')

    # Fit brute force nearest neighbors using FAISS
    print("\nFitting brute force nearest neighbors (FAISS)...")
    max_knn = max(knns)
    algo = BruteForceFAISS(metric="euclidean", dimension=num_dimensions)
    algo.fit(train)

    # Query for ground truth (batch processing)
    print(f"Querying for ground truth (k={max_knn})...")
    distances, indices = algo.query_batch(test, k=max_knn)

    # Save ground truth for each k value
    gt_path = get_ground_truth_path(dataset)

    print(f"\nSaving ground truth for k={max_knn} to: {gt_path}")

    # Create ground truth dictionary (query_idx -> list of neighbor indices)
    gt_dict = {}
    for i in range(num_queries):
        gt_dict[i] = indices[i, :max_knn].tolist()

    # Save to JSON
    with open(gt_path, 'w') as f:
        json.dump(gt_dict, f)

    print(f"Saved ground truth for {num_queries} queries with k={max_knn}")

    print(f"\n=== Ground truth generation complete for {dataset} ===")


def main():
    """Generate ground truth for the cohere dataset."""
    dataset = "fmnist"

    print("=" * 80)
    print("Ground Truth Generation")
    print("=" * 80)

    generate_ground_truth(dataset, knns=[100])

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
