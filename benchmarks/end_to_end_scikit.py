import os
threads = "10"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["BLIS_NUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads


from sklearn.cluster import KMeans
import numpy as np
import os
import time
import math
import sys
from bench_utils import DATASET_PARAMS, load_ground_truth, compute_recall, print_recall_results, KNN_VALUES

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "openai"

    if dataset not in DATASET_PARAMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(DATASET_PARAMS.keys())}"
        )
    num_vectors, num_dimensions = DATASET_PARAMS[dataset]
    num_centroids = max(1, int(math.sqrt(num_vectors) * 4))
    threads = 10
    n_iter = 2

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

    start = time.time()
    km = KMeans(
        n_clusters=num_centroids,
        init='random',
        n_init=1,
        max_iter=n_iter,
        verbose=1,
        random_state=42,
        copy_x=True
    )
    km.fit(data)
    end = time.time()
    print(f"ScikitLearn Took: {(end - start):.2f} s")

    # Compute recall if ground truth file exists
    gt_filename = f"{dataset}.json"
    queries_filename = f"data_{dataset}_test.bin"

    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Computing Recall ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")

        # Load ground truth
        gt_dict = load_ground_truth(gt_filename)

        # Load query vectors
        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = len(gt_dict)
        queries = queries.reshape(n_queries, num_dimensions)
        print(f"Loaded {n_queries} queries")

        # Get sklearn assignments (cluster labels for each data point)
        assignments = km.labels_  # shape: (num_vectors,)
        centroids = km.cluster_centers_  # shape: (num_centroids, num_dimensions)

        # Compute recall for different KNN values
        for knn in KNN_VALUES:
            results = compute_recall(gt_dict, assignments, queries, centroids, num_centroids, knn)
            print_recall_results(results, knn)
    else:
        if not os.path.exists(gt_filename):
            print(f"\nGround truth file not found: {gt_filename}")
        if not os.path.exists(queries_filename):
            print(f"Queries file not found: {queries_filename}")
