"""Utility functions for K-means benchmarking."""

import json
import numpy as np


# Dataset configurations: name -> (num_vectors, num_dimensions)
DATASET_PARAMS = {
    "mxbai": (769_382, 1024),
    "openai": (999_000, 1536),
    "arxiv": (2_253_000, 768),
    "sift": (1_000_000, 128),
    "fmnist": (60_000, 784),
    "glove100": (1_183_514, 100),
    "glove50": (1_183_514, 50),
    "gist": (1000000, 960),
    "contriever": (990000, 768)
}


# Standard exploration fractions for recall computation
EXPLORE_FRACTIONS = [
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.0100, 0.0125, 0.0150, 0.0175, 0.0200, 0.0225, 0.0250, 0.0275,
    0.0300, 0.0325, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450, 0.0475, 0.0500,
    0.1
]


# KNN values to test
KNN_VALUES = [10, 100]


def load_ground_truth(filename):
    """Load ground truth from JSON file.

    Args:
        filename: Path to JSON file containing ground truth

    Returns:
        Dictionary mapping query index (as string) to list of vector IDs
    """
    with open(filename, 'r') as f:
        return json.load(f)


def compute_recall(gt_dict, assignments, queries, centroids, num_centroids, knn):
    """Compute recall@K for different exploration fractions.

    Args:
        gt_dict: Ground truth dictionary (query_idx -> vector_ids)
        assignments: Cluster assignments for all data vectors
        queries: Query vectors (n_queries, d)
        centroids: Centroid vectors (num_centroids, d)
        num_centroids: Number of centroids
        knn: Number of ground truth neighbors to consider

    Returns:
        List of tuples (centroids_to_explore, explore_fraction, recall, avg_vectors_to_visit)
    """
    n_queries = len(gt_dict)

    # Count cluster sizes to compute vectors to visit
    cluster_sizes = np.bincount(assignments, minlength=num_centroids)

    # Compute distances from queries to centroids
    # Using L2 distance: ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
    query_norms = np.sum(queries ** 2, axis=1, keepdims=True)  # (n_queries, 1)
    centroid_norms = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, num_centroids)
    dot_products = queries @ centroids.T  # (n_queries, num_centroids)
    distances = query_norms + centroid_norms - 2 * dot_products  # (n_queries, num_centroids)

    results = []
    for explore_frac in EXPLORE_FRACTIONS:
        centroids_to_explore = max(1, int(num_centroids * explore_frac))

        # Find top-N nearest centroids for each query
        top_centroid_indices = np.argsort(distances, axis=1)[:, :centroids_to_explore]

        # Compute recall and vectors to visit
        total_recall = 0.0
        total_vectors_to_visit = 0
        for query_idx in range(n_queries):
            query_key = str(query_idx)
            if query_key not in gt_dict:
                continue

            gt_vector_ids = gt_dict[query_key][:knn]  # List of ground truth vector IDs
            top_centroids_list = top_centroid_indices[query_idx]  # Top N centroids for this query
            top_centroids = set(top_centroids_list)

            # Count vectors to visit for this query
            vectors_to_visit = np.sum(cluster_sizes[top_centroids_list])
            total_vectors_to_visit += vectors_to_visit

            # Count how many ground truth vectors have their assigned centroid in the top N
            found = 0
            for vector_id in gt_vector_ids:
                assigned_centroid = assignments[vector_id]  # Which centroid this vector belongs to
                if assigned_centroid in top_centroids:
                    found += 1

            query_recall = found / len(gt_vector_ids)
            total_recall += query_recall

        average_recall = total_recall / n_queries
        avg_vectors_to_visit = total_vectors_to_visit / n_queries
        results.append((centroids_to_explore, explore_frac, average_recall, avg_vectors_to_visit))

    return results


def print_recall_results(results, knn):
    """Print recall results in a formatted table.

    Args:
        results: List of tuples (centroids_to_explore, explore_fraction, recall, avg_vectors_to_visit)
        knn: KNN value used for this result set
    """
    print(f"\n--- Recall@{knn} ---")
    for centroids_to_explore, explore_frac, recall, avg_vectors in results:
        print(f"Recall@{centroids_to_explore:4d} ({explore_frac*100:5.2f}% centroids, {avg_vectors:8.0f} avg vectors): {recall:.4f}")
