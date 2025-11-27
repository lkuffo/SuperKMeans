#ifndef SUPERKMEANS_BATCH_COMPUTER_H
#define SUPERKMEANS_BATCH_COMPUTER_H

#include <cstdint>
#include <cstdio>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.hpp"
#include <Eigen/Eigen/Dense>

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class BatchComputer {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::u8> {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::f32> {

    using distance_t = skmeans_distance_t<Quantization::f32>;
    using data_t = skmeans_value_t<Quantization::f32>;
    using norms_t = skmeans_value_t<Quantization::f32>;
    using knn_candidate_t = KNNCandidate<Quantization::f32>;
    using layout_t = PDXLayout<Quantization::f32, DistanceFunction::l2>;
    using MatrixR = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  public:
    static void Batch_XRowMajor_YColMajor(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf
    ) {
        Eigen::Map<const MatrixR> x_matrix(x, n_x, d);
        Eigen::Map<const MatrixC> y_matrix(y, d, n_y);

        Eigen::Map<MatrixR> distances_matrix(all_distances_buf, n_x, n_y);
        distances_matrix.noalias() = x_matrix * y_matrix;
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            const float norm_x_i = norms_x[i];
            float* row_p = distances_matrix.data() + i * n_y;
            for (size_t j = 0; j < n_y; ++j) {
                row_p[j] = -2.0f * row_p[j] + norm_x_i + norms_y[j];
            }
            uint32_t knn_idx;
            auto top_1 = distances_matrix.row(i).minCoeff(&knn_idx);
            out_distances[i] = std::max(0.0f, top_1);
            out_knn[i] = knn_idx;
        }
    };

    static void Batch_XRowMajor_YRowMajor(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf
    ) {
        Eigen::Map<const MatrixR> x_matrix(x, n_x, d);
        Eigen::Map<const MatrixR> y_matrix(y, n_y, d); // YRowMajor

        Eigen::Map<MatrixR> distances_matrix(all_distances_buf, n_x, n_y);
        TicToc tt;
        tt.Reset();
        tt.Tic();
        distances_matrix.noalias() = x_matrix * y_matrix.transpose(); // YRowMajor
        tt.Toc();
        // std::cout << "Total time for BLAS multiplication (s): " << tt.accum_time / 1000000000.0
        //           << std::endl;
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            const float norm_x_i = norms_x[i];
            float* row_p = distances_matrix.data() + i * n_y;
            for (size_t j = 0; j < n_y; ++j) {
                row_p[j] = -2.0f * row_p[j] + norm_x_i + norms_y[j];
            }
            uint32_t knn_idx;
            auto top_1 = distances_matrix.row(i).minCoeff(&knn_idx);
            out_distances[i] = std::max(0.0f, top_1);
            out_knn[i] = knn_idx;
        }
    };

    static void Batched_XRowMajor_YRowMajor(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf
    ) {
        SKM_PROFILE_SCOPE("search");
        SKM_PROFILE_SCOPE("search/1st_blas");
        std::fill_n(out_distances, n_x, std::numeric_limits<distance_t>::max());
        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            auto batch_n_x = X_BATCH_SIZE;
            auto batch_x_p = x + (i * d);
            if (i + X_BATCH_SIZE > n_x) {
                batch_n_x = n_x - i;
            }
            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                auto batch_n_y = Y_BATCH_SIZE;
                auto batch_y_p = y + (j * d);
                if (j + Y_BATCH_SIZE > n_y) {
                    batch_n_y = n_y - j;
                }
                Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
                Eigen::Map<const MatrixR> x_matrix(batch_x_p, batch_n_x, d);
                Eigen::Map<const MatrixR> y_matrix(batch_y_p, batch_n_y, d);
                distances_matrix.noalias() = x_matrix * y_matrix.transpose();
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    const float norm_x_i = norms_x[i_idx];
                    float* row_p = distances_matrix.data() + r * batch_n_y;
#pragma clang loop vectorize(enable)
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                    }
                    uint32_t knn_idx;
                    auto batch_top_1 = distances_matrix.row(r).minCoeff(&knn_idx);
                    if (batch_top_1 < out_distances[i_idx]) {
                        out_distances[i_idx] = std::max(0.0f, batch_top_1);
                        out_knn[i_idx] = j + knn_idx;
                    }
                }
            }
        }
    };

    // Batched Y version: Y is processed in batches, need to merge top-k across batches
    static void Batched_XRowMajor_YRowMajor_TopK(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        const size_t k,
        uint32_t* SKM_RESTRICT out_knn,      // Size: n_x * k
        distance_t* SKM_RESTRICT out_distances, // Size: n_x * k
        float* SKM_RESTRICT all_distances_buf
    ) {
        // Initialize output with infinity
        std::fill_n(out_distances, n_x * k, std::numeric_limits<distance_t>::max());
        std::fill_n(out_knn, n_x * k, static_cast<uint32_t>(-1));

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            auto batch_n_x = X_BATCH_SIZE;
            auto batch_x_p = x + (i * d);
            if (i + X_BATCH_SIZE > n_x) {
                batch_n_x = n_x - i;
            }
            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                auto batch_n_y = Y_BATCH_SIZE;
                auto batch_y_p = y + (j * d);
                if (j + Y_BATCH_SIZE > n_y) {
                    batch_n_y = n_y - j;
                }

                Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
                Eigen::Map<const MatrixR> x_matrix(batch_x_p, batch_n_x, d);
                Eigen::Map<const MatrixR> y_matrix(batch_y_p, batch_n_y, d);
                distances_matrix.noalias() = x_matrix * y_matrix.transpose();

#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    const float norm_x_i = norms_x[i_idx];
                    float* row_p = distances_matrix.data() + r * batch_n_y;

                    // Compute L2 distances for current batch
#pragma clang loop vectorize(enable)
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                    }

                    // Merge: Combine previous top-k with current Y batch candidates
                    std::vector<std::pair<float, uint32_t>> candidates;
                    candidates.reserve(k + batch_n_y);

                    // Add previous top-k (skip if distance is infinity, meaning not filled yet)
                    for (size_t ki = 0; ki < k; ++ki) {
                        if (out_distances[i_idx * k + ki] < std::numeric_limits<distance_t>::max()) {
                            candidates.push_back({out_distances[i_idx * k + ki], out_knn[i_idx * k + ki]});
                        }
                    }

                    // Add current batch candidates
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        candidates.push_back({row_p[c], static_cast<uint32_t>(j + c)});
                    }

                    // Partial sort to get new top-k
                    size_t actual_k = std::min(k, candidates.size());
                    std::partial_sort(
                        candidates.begin(),
                        candidates.begin() + actual_k,
                        candidates.end()
                    );

                    // Update output with new top-k
                    for (size_t ki = 0; ki < actual_k; ++ki) {
                        out_distances[i_idx * k + ki] = std::max(0.0f, candidates[ki].first);
                        out_knn[i_idx * k + ki] = candidates[ki].second;
                    }

                    // Fill remaining slots if needed
                    for (size_t ki = actual_k; ki < k; ++ki) {
                        out_distances[i_idx * k + ki] = std::numeric_limits<distance_t>::max();
                        out_knn[i_idx * k + ki] = static_cast<uint32_t>(-1);
                    }
                }
            }
        }
    };

    static void Batched_XRowMajor_YRowMajor_PartialD(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y, // Full-dimensional centroids (only partial_d dims used for BLAS)
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf,
        const layout_t& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts = nullptr
    ) {
        SKM_PROFILE_SCOPE("search");
        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            auto batch_n_x = X_BATCH_SIZE;
            auto batch_x_p = x + (i * d);
            if (i + X_BATCH_SIZE > n_x) {
                batch_n_x = n_x - i;
            }
            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                auto batch_n_y = Y_BATCH_SIZE;
                auto batch_y_p = y + (j * d);
                if (j + Y_BATCH_SIZE > n_y) {
                    batch_n_y = n_y - j;
                }
                Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
                {
                    SKM_PROFILE_SCOPE("search/blas");
                    Eigen::Map<const MatrixR> x_matrix(batch_x_p, batch_n_x, d);
                    Eigen::Map<const MatrixR> y_matrix(batch_y_p, batch_n_y, d);
                    distances_matrix.noalias() =
                        x_matrix.leftCols(partial_d) * y_matrix.leftCols(partial_d).transpose();
                }
                {
                    SKM_PROFILE_SCOPE("search/norms");
#pragma omp parallel for num_threads(g_n_threads)
                    for (size_t r = 0; r < batch_n_x; ++r) {
                        const auto i_idx = i + r;
                        const float norm_x_i = norms_x[i_idx];
                        float* row_p = distances_matrix.data() + r * batch_n_y;
#pragma clang loop vectorize(enable)
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                        }
                    }
                }
                {
                    SKM_PROFILE_SCOPE("search/pdx");
#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
                    for (size_t r = 0; r < batch_n_x; ++r) {
                        const auto i_idx = i + r;
                        auto data_p = x + (i_idx * d);
                        // Note that this will take the KNN from the previous batch loop
                        const auto prev_assignment = out_knn[i_idx];
                        distance_t dist_to_prev_centroid;
                        if (j == 0) { // After this we always have the right distance in out_distances
                            dist_to_prev_centroid = DistanceComputer<DistanceFunction::l2, Quantization::f32>::Horizontal(
                                y + (prev_assignment * d), data_p, d
                            );
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        // PDXearch per vector
                        knn_candidate_t assignment;
                        auto partial_distances_p = distances_matrix.data() + r * batch_n_y;
                        size_t local_not_pruned = 0;
                        assignment =
                            pdx_centroids.searcher
                                ->Top1PartialSearchWithThresholdAndPartialDistances(
                                    data_p,
                                    dist_to_prev_centroid,
                                    prev_assignment,
                                    partial_distances_p,
                                    partial_d,
                                    j / VECTOR_CHUNK_SIZE, // start cluster_id
                                    (j + Y_BATCH_SIZE) /
                                        VECTOR_CHUNK_SIZE, // end cluster_id; We use Y_BATCH_SIZE and
                                                          // not batch_n_y because otherwise we
                                                          // would not go up until incomplete
                                                          // clusters
                                    out_not_pruned_counts != nullptr ? &local_not_pruned : nullptr
                                );
                        // Store not-pruned count for this X vector (accumulate across Y batches)
                        if (out_not_pruned_counts != nullptr) {
                            out_not_pruned_counts[i_idx] += local_not_pruned;
                        }
                        auto [assignment_idx, assignment_distance] = assignment;
                        out_knn[i_idx] = assignment_idx;
                        out_distances[i_idx] = assignment_distance;
                    }
                }
            }
        }
    }
};

} // namespace skmeans

#endif // SUPERKMEANS_BATCH_COMPUTER_H
