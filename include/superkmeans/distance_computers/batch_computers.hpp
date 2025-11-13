#ifndef SUPERKMEANS_BATCH_COMPUTER_H
#define SUPERKMEANS_BATCH_COMPUTER_H

#include <cstdint>
#include <cstdio>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/pdx/layout.h"
#include <Eigen/Eigen/Dense>

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class BatchComputer {};

template <>
class BatchComputer<l2, u8> {};

template <>
class BatchComputer<l2, f32> {

    static constexpr size_t serial_threshold = 20;
    using distance_t = skmeans_distance_t<f32>;
    using data_t = skmeans_value_t<f32>;
    using norms_t = skmeans_value_t<f32>;
    using knn_candidate_t = KNNCandidate<f32>;
    using layout_t = PDXLayout<f32, l2>;
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
        // Eigen::Map<const MatrixR> y_matrix(y, n_y, d); // YRowMajor

        Eigen::Map<MatrixR> distances_matrix(all_distances_buf, n_x, n_y);
        distances_matrix.noalias() = x_matrix * y_matrix;
        // distances_matrix.noalias() = x_matrix * y_matrix.transpose(); // YRowMajor
#pragma omp parallel for num_threads(14)
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
#pragma omp parallel for num_threads(14)
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
        std::fill_n(out_distances, n_x, std::numeric_limits<distance_t>::max());
        TicToc tt;
        tt.Reset();
        tt.Tic();
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
                // std::cout << batch_n_x << "," << batch_n_y << std::endl;
                Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
                Eigen::Map<const MatrixR> x_matrix(batch_x_p, batch_n_x, d);
                Eigen::Map<const MatrixR> y_matrix(batch_y_p, batch_n_y, d);  // YRowMajor
                distances_matrix.noalias() = x_matrix * y_matrix.transpose(); // YRowMajor
#pragma omp parallel for num_threads(14)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    const float norm_x_i = norms_x[i_idx];
                    float* row_p = distances_matrix.data() + r * batch_n_y;
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
        tt.Toc();
        std::cout << "Total time for BLAS multiplication (s): " << tt.accum_time / 1000000000.0
                  << std::endl;
    };

    static void Batched_XRowMajor_YRowMajor_PartialD(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const data_t* SKM_RESTRICT prev_y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        float* SKM_RESTRICT all_distances_buf,
        const layout_t& pdx_centroids,
        const uint32_t partial_d
    ) {
        TicToc tt;
        tt.Reset();
        tt.Tic();
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
                distances_matrix.noalias() =
                    x_matrix.leftCols(partial_d) * y_matrix.leftCols(partial_d).transpose();
#pragma omp parallel for num_threads(14)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    const float norm_x_i = norms_x[i_idx];
                    float* row_p = distances_matrix.data() + r * batch_n_y;
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                    }
                }
#pragma omp parallel for num_threads(14)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    auto data_p = x + (i_idx * d);
                    const auto prev_assignment = out_knn[i_idx];
                    const distance_t dist_to_prev_centroid = DistanceComputer<l2, f32>::Horizontal(
                        prev_y + (prev_assignment * d), data_p, d
                    );
                    // PDXearch per vector
                    auto partial_distances_p = distances_matrix.data() + r * batch_n_y;
                    std::vector<knn_candidate_t> assignment =
                        pdx_centroids.searcher->Top1PartialSearchWithThresholdAndPartialDistances(
                            data_p,
                            dist_to_prev_centroid,
                            prev_assignment,
                            r,
                            partial_distances_p,
                            partial_d,
                            j / VECTOR_CHUNK_SIZE,              // start cluster_id
                            (j + batch_n_y) / VECTOR_CHUNK_SIZE // end cluster_id
                        );
                    auto [assignment_idx, assignment_distance] = assignment[0];
                    out_knn[i_idx] = assignment_idx;
                    out_distances[i_idx] = assignment_distance;
                }
            }
        }
        tt.Toc();
        std::cout << "Total time for BLAS+PDX (s): " << tt.accum_time / 1000000000.0 << std::endl;
    };

    static void Batch_XRowMajor_YRowMajor_PartialD(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        const norms_t* SKM_RESTRICT norms_x,
        const norms_t* SKM_RESTRICT norms_y,
        float* SKM_RESTRICT all_distances_buf,
        const uint32_t partial_d
    ) {
        Eigen::Map<const MatrixR> x_matrix(x, n_x, d);
        Eigen::Map<const MatrixR> y_matrix(y, n_y, d); // YRowMajor

        Eigen::Map<MatrixR> distances_matrix(all_distances_buf, n_x, n_y);
        TicToc tt;
        tt.Reset();
        tt.Tic();
        distances_matrix.noalias() =
            x_matrix.leftCols(partial_d) * y_matrix.leftCols(partial_d).transpose(); // YRowMajor
        tt.Toc();
        // std::cout << "Total time for BLAS multiplication (s): " << tt.accum_time / 1000000000.0
        //           << std::endl;
#pragma omp parallel for num_threads(14)
        for (size_t i = 0; i < n_x; ++i) {
            const float norm_x_i = norms_x[i];
            float* row_p = distances_matrix.data() + i * n_y;
            for (size_t j = 0; j < n_y; ++j) {
                row_p[j] = -2.0f * row_p[j] + norm_x_i + norms_y[j];
            }
        }
    };

    static void Batch_XRowMajor_YColMajor_Normalized(
        const data_t* SKM_RESTRICT x,
        const data_t* SKM_RESTRICT y,
        const size_t n_x,
        const size_t n_y,
        const size_t d,
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances
    ) {
        std::vector<float> all_distances(n_x * n_y);
        Eigen::Map<const MatrixR> x_matrix(x, n_x, d);
        Eigen::Map<const MatrixC> y_matrix(y, d, n_y);
        Eigen::Map<MatrixR> distances_matrix(all_distances.data(), n_x, n_y);
        distances_matrix.noalias() = x_matrix * y_matrix;
#pragma omp parallel for num_threads(14)
        for (size_t i = 0; i < n_x; ++i) {
            float* row_p = distances_matrix.data() + i * n_y;
            for (size_t j = 0; j < n_y; ++j) {
                row_p[j] = 2.0f - 2.0f * row_p[j];
            }
            uint32_t knn_idx;
            auto top_1 = distances_matrix.row(i).minCoeff(&knn_idx);
            out_distances[i] = std::max(0.0f, top_1);
            out_knn[i] = knn_idx;
        }
    };

  private:
    void GetTop1FromL2DistancesMatrix() {
        // Just to modularize that last part...
    }
};

} // namespace skmeans

#endif // SUPERKMEANS_BATCH_COMPUTER_H
