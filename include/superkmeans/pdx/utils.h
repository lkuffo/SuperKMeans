#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace skmeans {

class TicToc {
  public:
    size_t accum_time = 0;
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    void Reset() {
        accum_time = 0;
        start = std::chrono::high_resolution_clock::now();
    }

    void Tic() { start = std::chrono::high_resolution_clock::now(); }

    void Toc() {
        auto end = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

inline uint32_t CeilXToMultipleOfM(uint32_t x, uint32_t m) {
    return (m == 0) ? x : ((x + m - 1) / m) * m;
}

inline uint32_t FloorXToMultipleOfM(uint32_t x, uint32_t m) {
    return (m == 0) ? x : (x / m) * m;
}

inline bool IsPowerOf2(const uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

/**
 * @brief Generate synthetic clusterable data (scikit-learn style make_blobs)
 *
 * Creates n_samples data points distributed around n_centers cluster centers.
 * Each center is randomly generated, and points are sampled from a Gaussian
 * distribution around their assigned center.
 *
 * @param n_samples Number of samples to generate
 * @param n_features Dimensionality of each sample
 * @param n_centers Number of cluster centers
 * @param normalize If true, L2-normalize each vector to unit length
 * @param cluster_std Standard deviation of points around centers
 * @param center_spread Standard deviation for generating cluster centers
 * @param random_state Seed for reproducibility
 * @return Flattened row-major vector of size n_samples * n_features
 */
inline std::vector<float> MakeBlobs(
    size_t n_samples,
    size_t n_features,
    size_t n_centers,
    bool normalize = false,
    float cluster_std = 1.0f,
    float center_spread = 10.0f,
    uint32_t random_state = 42
) {
    std::mt19937 gen(random_state);
    std::normal_distribution<float> center_dist(0.0f, center_spread);
    std::vector<std::vector<float>> centers(n_centers, std::vector<float>(n_features));
    for (auto& c : centers) {
        for (auto& x : c) {
            x = center_dist(gen);
        }
    }
    std::uniform_int_distribution<size_t> cluster_dist(0, n_centers - 1);
    std::normal_distribution<float> point_dist(0.0f, cluster_std);
    std::vector<float> data;
    data.reserve(n_samples * n_features);
    for (size_t i = 0; i < n_samples; ++i) {
        const auto& center = centers[cluster_dist(gen)];
        for (size_t j = 0; j < n_features; ++j) {
            data.push_back(center[j] + point_dist(gen));
        }
    }

    if (normalize) {
        for (size_t i = 0; i < n_samples; ++i) {
            float norm_sq = 0.0f;
            for (size_t j = 0; j < n_features; ++j) {
                norm_sq += data[i * n_features + j] * data[i * n_features + j];
            }
            float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (size_t j = 0; j < n_features; ++j) {
                data[i * n_features + j] *= inv_norm;
            }
        }
    }

    return data;
}

/**
 * @brief Generate random vectors with uniform distribution
 *
 * Creates n vectors of d dimensions with values uniformly distributed
 * in the range [min_val, max_val].
 *
 * @param n Number of vectors to generate
 * @param d Dimensionality of each vector
 * @param min_val Minimum value (default: -1.0f)
 * @param max_val Maximum value (default: 1.0f)
 * @param seed Random seed for reproducibility (default: 42)
 * @return Flattened row-major vector of size n * d
 */
inline std::vector<float> GenerateRandomVectors(
    size_t n,
    size_t d,
    float min_val = -1.0f,
    float max_val = 1.0f,
    uint32_t seed = 42
) {
    std::vector<float> output(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& val : output) {
        val = dist(rng);
    }
    return output;
}

// ============================================================================
// Testing Utilities
// The following functions are brute-force reference implementations used
// exclusively in the testing suite for correctness verification.
// ============================================================================

/**
 * @brief Compute squared L2 distance between two vectors (testing only)
 *
 * @param a First vector (d elements)
 * @param b Second vector (d elements)
 * @param d Dimensionality
 * @return Squared L2 distance
 */
inline float ComputeL2DistanceSquared(const float* a, const float* b, size_t d) {
    float dist = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief Compute squared L2 norms for row-major vectors (testing only)
 *
 * @param data Pointer to row-major data (n × d)
 * @param n Number of vectors
 * @param d Dimensionality
 * @return Vector of squared L2 norms
 */
inline std::vector<float> ComputeNorms(const float* data, size_t n, size_t d) {
    std::vector<float> norms(n);
    for (size_t i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            norm += data[i * d + j] * data[i * d + j];
        }
        norms[i] = norm;
    }
    return norms;
}

/**
 * @brief Find nearest neighbor for each query using brute force (testing only)
 *
 * @param x Query vectors (n_x × d, row-major)
 * @param y Candidate vectors (n_y × d, row-major)
 * @param n_x Number of query vectors
 * @param n_y Number of candidate vectors
 * @param d Dimensionality
 * @param out_knn Output: index of nearest neighbor for each query (size: n_x)
 * @param out_distances Output: distance to nearest neighbor (size: n_x)
 */
inline void FindNearestNeighborBruteForce(
    const float* x,
    const float* y,
    size_t n_x,
    size_t n_y,
    size_t d,
    uint32_t* out_knn,
    float* out_distances
) {
    for (size_t i = 0; i < n_x; ++i) {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best_idx = 0;
        for (size_t j = 0; j < n_y; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = x[i * d + k] - y[j * d + k];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<uint32_t>(j);
            }
        }
        out_knn[i] = best_idx;
        out_distances[i] = best_dist;
    }
}

/**
 * @brief Find k nearest neighbors for each query using brute force (testing only)
 *
 * @param x Query vectors (n_x × d, row-major)
 * @param y Candidate vectors (n_y × d, row-major)
 * @param n_x Number of query vectors
 * @param n_y Number of candidate vectors
 * @param d Dimensionality
 * @param k Number of nearest neighbors to find
 * @param out_knn Output: indices of k nearest neighbors (size: n_x × k)
 * @param out_distances Output: distances to k nearest neighbors (size: n_x × k)
 */
inline void FindKNearestNeighborsBruteForce(
    const float* x,
    const float* y,
    size_t n_x,
    size_t n_y,
    size_t d,
    size_t k,
    uint32_t* out_knn,
    float* out_distances
) {
    std::vector<std::pair<float, uint32_t>> distances(n_y);
    for (size_t i = 0; i < n_x; ++i) {
        for (size_t j = 0; j < n_y; ++j) {
            float dist = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                float diff = x[i * d + dim] - y[j * d + dim];
                dist += diff * diff;
            }
            distances[j] = {dist, static_cast<uint32_t>(j)};
        }
        size_t actual_k = std::min(k, n_y);
        std::partial_sort(distances.begin(), distances.begin() + actual_k, distances.end());
        for (size_t ki = 0; ki < actual_k; ++ki) {
            out_knn[i * k + ki] = distances[ki].second;
            out_distances[i * k + ki] = distances[ki].first;
        }
    }
}

/**
 * @brief Find nearest centroid for a single point using brute force (testing only)
 *
 * @param point Single query vector (d elements)
 * @param centroids Centroid vectors (n_clusters × d, row-major)
 * @param n_clusters Number of centroids
 * @param d Dimensionality
 * @return Index of the nearest centroid
 */
inline uint32_t FindNearestCentroidBruteForce(
    const float* point,
    const float* centroids,
    size_t n_clusters,
    size_t d
) {
    uint32_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t c = 0; c < n_clusters; ++c) {
        float dist = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float diff = point[j] - centroids[c * d + j];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<uint32_t>(c);
        }
    }
    return best_idx;
}

} // namespace skmeans