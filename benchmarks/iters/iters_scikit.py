import os
import sys

# Add parent directory to path for bench_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["BLIS_NUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)

from sklearn.cluster import KMeans
import numpy as np
import time
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         N_QUERIES, ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

if __name__ == "__main__":
    algorithm = "scikit"
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fmnist"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "iters_scikit"
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = get_default_n_clusters(num_vectors)
    threads = threads

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}")

    filename = get_data_path(dataset)
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)
    if dataset in ANGULAR_DATASETS:
        print(f"\nWARNING: Dataset '{dataset}' should use spherical k-means, "
              f"but scikit-learn does not support this. Results may be suboptimal.")

    gt_filename = get_ground_truth_path(dataset)
    queries_filename = get_query_path(dataset)
    gt_dict = None
    queries = None
    n_queries = 0
    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Loading Ground Truth and Queries ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")

        gt_dict = load_ground_truth(gt_filename)
        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = N_QUERIES
        queries = queries[:n_queries * num_dimensions].reshape(n_queries, num_dimensions)
        print(f"Loaded {n_queries} queries and {len(gt_dict)} ground truth entries")
    else:
        if not os.path.exists(gt_filename):
            print(f"\nGround truth file not found: {gt_filename}")
        if not os.path.exists(queries_filename):
            print(f"Queries file not found: {queries_filename}")
        print("Recall computation will be skipped")

    init_methods = ['random', 'k-means++']
    iteration_counts = list(range(1, 11))
    for init_method in init_methods:
        print("\n##########################################")
        print(f"# init = {init_method}")
        print("##########################################")

        # For k-means++, we'll do incremental training to avoid re-running initialization
        previous_centers = None
        cumulative_time_ms = 0.0

        for n_iter in iteration_counts:
            print("\n========================================")
            print(f"Running with init = {init_method}, target iterations = {n_iter}")
            print("========================================")

            if init_method == 'k-means++' and previous_centers is not None:
                # Resume from previous iteration - run just 1 more iteration
                actual_init = previous_centers
                actual_max_iter = 1
                print(f"Resuming from previous iteration (running 1 more iteration)")
            else:
                actual_init = init_method
                actual_max_iter = n_iter
                cumulative_time_ms = 0.0  # Reset for 'random' or first k-means++ run

            km = KMeans(
                n_clusters=num_centroids,
                init=actual_init,
                n_init=1,
                max_iter=actual_max_iter,
                tol=0.0,  # We dont want early stopping
                verbose=0,
                random_state=42,
                copy_x=True
            )


            with Timer() as timer:
                km.fit(data)
            construction_time_ms = timer.get_milliseconds()
            actual_iterations = km.n_iter_
            final_objective = km.inertia_

            # For k-means++, accumulate time and track total iterations
            if init_method == 'k-means++':
                cumulative_time_ms += construction_time_ms
                total_iterations = n_iter
                # Store centers for next iteration
                previous_centers = km.cluster_centers_.copy()
            else:
                # For 'random', each run is independent
                total_iterations = actual_iterations
                cumulative_time_ms = construction_time_ms

            print(f"\nThis step completed in {construction_time_ms:.2f} ms")
            print(f"This step iterations: {actual_iterations}")
            if init_method == 'k-means++':
                print(f"Cumulative time: {cumulative_time_ms:.2f} ms")
                print(f"Total iterations so far: {total_iterations}")
            print(f"Final objective (inertia): {final_objective}")

            if gt_dict is not None and queries is not None:
                print(f"\n--- Computing Recall ---")

                assignments = km.labels_
                centroids = km.cluster_centers_

                results_knn_10 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 10)
                print_recall_results(results_knn_10, 10)
                results_knn_100 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 100)
                print_recall_results(results_knn_100, 100)

                config_dict = {
                    "init": init_method,  # Use original init method, not km.init
                    "n_init": str(km.n_init),
                    "max_iter": str(n_iter),  # Use target iterations, not actual_max_iter
                    "random_state": str(km.random_state),
                    "copy_x": str(km.copy_x).lower(),
                    "tol": str(km.tol),
                    "algorithm": str(km.algorithm)
                }

                write_results_to_csv(
                    experiment_name, algorithm, dataset, n_iter, total_iterations,
                    num_dimensions, num_vectors, num_centroids, cumulative_time_ms,
                    threads, final_objective, config_dict,
                    results_knn_10, results_knn_100
                )
            else:
                print("Skipping CSV output (recall computation requires ground truth)")

