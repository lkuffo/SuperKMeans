#ifndef SUPERKMEANS_BATCH_COMPUTER_H
#define SUPERKMEANS_BATCH_COMPUTER_H

#include <cstdint>
#include <cstdio>

#include "superkmeans/common.h"
#include <Eigen/Eigen/Dense>

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class BatchComputer {};

template <>
class BatchComputer<l2, u8> {};

template <>
class BatchComputer<l2, f32> {

    static constexpr size_t x_batch_size = 4096;
    static constexpr size_t y_batch_size = 1024;
    static constexpr size_t serial_threshold = 20;
    using distance_t = skmeans_distance_t<f32>;
    using data_t = skmeans_value_t<f32>;
    using norms_t = skmeans_value_t<f32>;
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
        distances_matrix.noalias() = x_matrix * y_matrix.transpose(); // YRowMajor
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
