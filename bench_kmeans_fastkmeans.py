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
import json
import fastkmeans
from fastkmeans import FastKMeans

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mxbai"
    dataset_params = {
        "mxbai": (769_382, 1024),
        "openai": (999_000, 1536),
        "arxiv": (2_253_000, 768),
        "sift": (1_000_000, 128),
        "fmnist": (60_000, 784),
        "glove100": (1_183_514, 100),
        "glove50": (1_183_514, 50)
    }

    if dataset not in dataset_params:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(dataset_params.keys())}"
        )
    num_vectors, num_dimensions = dataset_params[dataset]
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
    km = FastKMeans(
        d=num_dimensions,
        k=num_centroids,
        niter=n_iter,
        tol=-math.inf,
        device='cpu',
        seed=42,
        max_points_per_centroid=256,
        verbose=True,
        use_triton=False,
    )
    km.train(data)
    end = time.time()
    print(f"FastKMeans Took: {(end - start):.2f} s")

    centroids = km.centroids
    assignments = km.predict(data)

    # Compute recall if ground truth file exists
    gt_filename = f"agnews-{dataset}-{num_dimensions}-euclidean_10.json"
    queries_filename = f"data_{dataset}_test.bin"

    if os.path.exists(gt_filename) and os.path.exists(queries_filename):
        print(f"\n--- Computing Recall ---")
        print(f"Ground truth file: {gt_filename}")
        print(f"Queries file: {queries_filename}")

        # Load ground truth
        with open(gt_filename, 'r') as f:
            gt_dict = json.load(f)

        # Load query vectors
        queries = np.fromfile(queries_filename, dtype=np.float32)
        n_queries = len(gt_dict)
        queries = queries.reshape(n_queries, num_dimensions)
        print(f"Loaded {n_queries} queries")

        # Compute distances from queries to centroids
        # Using L2 distance: ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
        query_norms = np.sum(queries ** 2, axis=1, keepdims=True)  # (n_queries, 1)
        centroid_norms = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, num_centroids)
        dot_products = queries @ centroids.T  # (n_queries, num_centroids)
        distances = query_norms + centroid_norms - 2 * dot_products  # (n_queries, num_centroids)

        # Test different numbers of centroids to explore
        explore_fractions = [
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250, 0.0275,
            0.0300, 0.0325, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450, 0.0475, 0.0500,
            0.1
        ]
        KNN = 10
        for explore_frac in explore_fractions:
            centroids_to_explore = max(1, int(num_centroids * explore_frac))

            # Find top-N nearest centroids for each query
            top_centroid_indices = np.argsort(distances, axis=1)[:, :centroids_to_explore]  # (n_queries, centroids_to_explore)

            # Compute recall
            total_recall = 0.0
            for query_idx in range(n_queries):
                query_key = str(query_idx)
                if query_key not in gt_dict:
                    continue

                gt_vector_ids = gt_dict[query_key][:KNN]  # List of ground truth vector IDs
                top_centroids = set(top_centroid_indices[query_idx])  # Top N centroids for this query

                # Count how many ground truth vectors have their assigned centroid in the top N
                found = 0
                for vector_id in gt_vector_ids:
                    assigned_centroid = assignments[vector_id]  # Which centroid this vector belongs to
                    if assigned_centroid in top_centroids:
                        found += 1

                query_recall = found / len(gt_vector_ids)
                total_recall += query_recall

            average_recall = total_recall / n_queries
            print(f"Recall@{centroids_to_explore:4d} ({explore_frac*100:5.2f}% of centroids): {average_recall:.4f}")
    else:
        if not os.path.exists(gt_filename):
            print(f"\nGround truth file not found: {gt_filename}")
        if not os.path.exists(queries_filename):
            print(f"Queries file not found: {queries_filename}")
