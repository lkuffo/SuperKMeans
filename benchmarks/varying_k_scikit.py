import os

threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["BLIS_NUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

from sklearn.cluster import KMeans
import numpy as np
import os
import time
import math
import sys
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         MAX_ITERS, N_QUERIES, VARYING_K_VALUES, ANGULAR_DATASETS,
                         get_data_path, get_query_path, get_ground_truth_path)

if __name__ == "__main__":
    # Experiment configuration
    algorithm = "scikit"
    experiment_name = "varying_k"

    dataset = sys.argv[1] if len(sys.argv) > 1 else "sift"

    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    n_iter = MAX_ITERS
    threads = threads

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"Experiment: {experiment_name}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"threads={threads}, n_iter={n_iter}")

    # Load data file (expects float32, row-major, n*d values)
    filename = get_data_path(dataset)
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)

    # Check if this dataset should use angular/spherical k-means
    if dataset in ANGULAR_DATASETS:
        print(f"\nWARNING: Dataset '{dataset}' should use spherical k-means, "
              f"but scikit-learn does not support this. Results may be suboptimal.")

    # Loop over different n_clusters values
    for num_centroids in VARYING_K_VALUES:
        print(f"\n========================================")
        print(f"n_clusters={num_centroids}")
        print(f"========================================")

        # Use incremental training strategy: run 1 iteration at a time
        previous_centers = None
        cumulative_time_ms = 0.0

        # Configure KMeans
        km = KMeans(
            n_clusters=num_centroids,
            init='random',
            n_init=1,
            max_iter=n_iter,
            tol=0.0,  # We dont want early stopping
            verbose=0,
            random_state=42,
            copy_x=True
        )

        # Time the training
        with Timer() as timer:
            km.fit(data)
        construction_time_ms = timer.get_milliseconds()

        # Get actual iterations and final objective
        actual_iterations = km.n_iter_
        final_objective = km.inertia_

        print(f"\nTraining completed in {construction_time_ms:.2f} ms")
        print(f"Actual iterations: {actual_iterations} (requested: {n_iter})")
        print(f"Final objective (inertia): {final_objective}")

        # Skip assignment and recall computation for this benchmark
        # We only care about training time and objective for the varying_k parameter sweep
        results_knn_10 = []   # Empty
        results_knn_100 = []  # Empty

        # Create config dictionary with scikit-learn parameters
        config_dict = {
            "init": "random",
            "n_init": str(km.n_init),
            "max_iter": str(n_iter),
            "random_state": str(km.random_state),
            "copy_x": str(km.copy_x).lower(),
            "tol": str(km.tol),
            "algorithm": str(km.algorithm)
        }

        # Write results to CSV
        write_results_to_csv(
            experiment_name, algorithm, dataset, n_iter, actual_iterations,
            num_dimensions, num_vectors, num_centroids, construction_time_ms,
            threads, final_objective, config_dict,
            results_knn_10, results_knn_100
        )
