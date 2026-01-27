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
                         MAX_ITERS, N_QUERIES, SCIKIT_EARLY_TERM_MAX_ITERS, SCIKIT_EARLY_TERM_TOL,
                         ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

if __name__ == "__main__":
    experiment_name = "early_termination"
    algorithm = "scikit"
    dataset = sys.argv[1] if len(sys.argv) > 1 else "openai"
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = get_default_n_clusters(num_vectors)
    n_iter = SCIKIT_EARLY_TERM_MAX_ITERS
    threads = threads

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_iter={n_iter}")

    # Load data file (expects float32, row-major, n*d values)
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

    km = KMeans(
        n_clusters=num_centroids,
        init='random',
        n_init=1,
        max_iter=n_iter,
        tol=SCIKIT_EARLY_TERM_TOL,
        verbose=0,
        random_state=42,
        copy_x=True
    )

    with Timer() as timer:
        km.fit(data)
    construction_time_ms = timer.get_milliseconds()
    actual_iterations = km.n_iter_
    final_objective = km.inertia_

    print(f"\nTraining completed in {construction_time_ms:.2f} ms")
    print(f"Actual iterations: {actual_iterations} (requested: {n_iter})")
    print(f"Final objective (inertia): {final_objective}")

    gt_filename = get_ground_truth_path(dataset)
    queries_filename = get_query_path(dataset)
    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Computing Recall ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")

        gt_dict = load_ground_truth(gt_filename)

        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = N_QUERIES
        queries = queries[:n_queries * num_dimensions].reshape(n_queries, num_dimensions)
        print(f"Using {n_queries} queries (loaded {len(gt_dict)} from ground truth)")

        assignments = km.labels_
        centroids = km.cluster_centers_ 

        results_knn_10 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 10)
        print_recall_results(results_knn_10, 10)
        results_knn_100 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 100)
        print_recall_results(results_knn_100, 100)

        config_dict = {
            "init": str(km.init),
            "n_init": str(km.n_init),
            "max_iter": str(km.max_iter),
            "random_state": str(km.random_state),
            "copy_x": str(km.copy_x).lower(),
            "tol": str(km.tol),
            "algorithm": str(km.algorithm)
        }

        write_results_to_csv(
            experiment_name, algorithm, dataset, n_iter, actual_iterations,
            num_dimensions, num_vectors, num_centroids, construction_time_ms,
            threads, final_objective, config_dict,
            results_knn_10, results_knn_100
        )
    else:
        if not os.path.exists(gt_filename):
            print(f"\nGround truth file not found: {gt_filename}")
        if not os.path.exists(queries_filename):
            print(f"Queries file not found: {queries_filename}")
        print("Skipping CSV output (recall computation requires ground truth)")
