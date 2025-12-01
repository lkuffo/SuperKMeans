import os
threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["BLIS_NUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)


import numpy as np
import math
import sys
from fastkmeans import FastKMeans
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         MAX_ITERS, N_QUERIES)

if __name__ == "__main__":
    # Experiment configuration
    experiment_name = "end_to_end"
    algorithm = "fastkmeans"

    dataset = sys.argv[1] if len(sys.argv) > 1 else "openai"

    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = max(1, int(math.sqrt(num_vectors) * 4))
    threads = threads
    n_iter = MAX_ITERS

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_iter={n_iter}")

    # Load data file (expects float32, row-major, n*d values)
    filename = f"data_{dataset}.bin"
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)

    # Configure FastKMeans
    km = FastKMeans(
        d=num_dimensions,
        k=num_centroids,
        niter=n_iter,
        tol=0.0,  # We don't want early stopping
        device='cpu',
        seed=42,
        max_points_per_centroid=99999, # We don't want early stopping
        verbose=False,
        use_triton=False,
    )

    # Time the training
    with Timer() as timer:
        km.train(data)
    construction_time_ms = timer.get_milliseconds()

    centroids = km.centroids
    assignments = km.predict(data)

    # Get actual iterations
    # Note: FastKMeans may not expose this directly, using requested value as fallback
    actual_iterations = getattr(km, 'n_iter_', n_iter)

    # Calculate objective: sum of squared L2 distances from each point to its assigned centroid
    final_objective = 0.0
    for i in range(num_vectors):
        centroid_idx = assignments[i]
        diff = data[i] - centroids[centroid_idx]
        final_objective += np.sum(diff ** 2)
    print(f"\nTraining completed in {construction_time_ms:.2f} ms")
    print(f"Actual iterations: {actual_iterations} (requested: {n_iter})")
    print(f"Final objective: {final_objective}")

    # Compute recall if ground truth file exists
    gt_filename = f"{dataset}.json"
    queries_filename = f"data_{dataset}_test.bin"

    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Computing Recall ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")

        # Load ground truth
        gt_dict = load_ground_truth(gt_filename)

        # Load query vectors (only first N_QUERIES)
        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = N_QUERIES
        queries = queries[:n_queries * num_dimensions].reshape(n_queries, num_dimensions)
        print(f"Using {n_queries} queries (loaded {len(gt_dict)} from ground truth)")

        # Compute recall for both KNN values
        results_knn_10 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 10)
        print_recall_results(results_knn_10, 10)

        results_knn_100 = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, 100)
        print_recall_results(results_knn_100, 100)

        # Create config dictionary with FastKMeans parameters
        config_dict = {
            "d": str(num_dimensions),
            "k": str(num_centroids),
            "niter": str(n_iter),
            "tol": str(km.tol),
            "device": str(km.device),
            "seed": str(km.seed),
            "max_points_per_centroid": str(km.max_points_per_centroid),
            "verbose": str(km.verbose).lower(),
            "use_triton": str(km.use_triton).lower()
        }

        # Write results to CSV
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
