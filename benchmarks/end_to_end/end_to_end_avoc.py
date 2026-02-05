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
import evoc
from bench_utils import (DATASET_PARAMS, load_ground_truth, compute_recall,
                         print_recall_results, KNN_VALUES, Timer, write_results_to_csv,
                         MAX_ITERS, N_QUERIES, ANGULAR_DATASETS, get_default_n_clusters,
                         get_data_path, get_query_path, get_ground_truth_path)

if __name__ == "__main__":
    algorithm = "scikit"
    dataset = sys.argv[1] if len(sys.argv) > 1 else "glove200"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "end_to_end"
    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = get_default_n_clusters(num_vectors)
    n_iter = MAX_ITERS
    threads = threads

    print(f"=== Running algorithm: {algorithm} ===")
    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_iter={n_iter}")

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

    clusterer = evoc.EVoC()

    print(f"Fitting data with EVoC")
    with Timer() as timer:
        clusterer.fit(
            data,
            approx_n_clusters=num_centroids,
            random_state=42,
            n_epochs=n_iter,
        )
    construction_time_ms = timer.get_milliseconds()
    print(f"Training completed in {construction_time_ms:.2f} ms")