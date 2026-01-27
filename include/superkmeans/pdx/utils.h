#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
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
inline std::vector<float> make_blobs(
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

} // namespace skmeans