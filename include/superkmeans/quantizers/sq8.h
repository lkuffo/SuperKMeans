#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <utility>
#include <vector>

#include <cpuinfo.h>
#include <Eigen/Dense>
#include <numkong/numkong.h>
#include "ruy/ruy.h"

namespace skmeans {

struct ScalarQuantizationParams {
    float quantization_base;
    float quantization_scale;
    float inv_quantization_scale;
};

/**
 * @brief 8-bit scalar quantizer with ruy GEMM backend.
 *
 * Global min/max quantization: q[i] = round((val[i] - base) * scale), clamped to [0, MAX_VALUE].
 * For L2 distance the base cancels: ||x-y||² = inv_scale² * Σ(x_q - y_q)².
 */
class SQ8Quantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;
    using u8_computer = DistanceComputer<DistanceFunction::l2, Quantization::u8>;

    static constexpr uint8_t MAX_VALUE = 255;

#if defined(__ARM_NEON)
    static constexpr bool is_arm = true;
#else
    static constexpr bool is_arm = false;
#endif

    SQ8Quantizer() {
        cpuinfo_initialize();
        has_amx = cpuinfo_has_x86_amx_int8();
    }

    /**
     * @brief u8×u8→u32 dot product GEMM, dispatching between NumKong and ruy.
     *
     * - ARM: always ruy (best for thin/wide matrices on NEON).
     * - x86 with AMX: always NumKong (leverages AMX tiles).
     * - x86 without AMX (e.g. Zen 5): NumKong when k > THIN_MATRIX_THRESHOLD,
     *   ruy otherwise (NumKong is subpar for small k).
     *
     * Handles OMP parallelization internally (backends need different threading).
     */
    void MatrixMultiplication(
        const quantized_t* a,
        const quantized_t* b,
        uint32_t* out,
        size_t m,
        size_t n,
        size_t k,
        size_t a_stride,
        size_t b_stride
    ) const {
        // NumKong u8 path: AMX or wide matrices on x86
        if (!is_arm && (has_amx || k > THIN_MATRIX_THRESHOLD)) {
            const size_t pack_size = nk_dots_packed_size_u8(n, k);
            if (pack_size > centroids_nk_packed_buf.size()) centroids_nk_packed_buf.resize(pack_size);
            nk_dots_pack_u8(b, n, k, b_stride, centroids_nk_packed_buf.data());

            const size_t c_stride = n * sizeof(uint32_t);

#pragma omp parallel num_threads(g_n_threads)
            {
                nk_configure_thread(nk_capabilities());
                int tid = omp_get_thread_num();
                int nt = omp_get_num_threads();
                size_t rows_per_t = (m + nt - 1) / nt;
                size_t start = tid * rows_per_t;
                size_t count = std::min(rows_per_t, m - start);
                if (start < m && count > 0) {
                    nk_dots_packed_u8(
                        a + start * a_stride,
                        centroids_nk_packed_buf.data(),
                        out + start * n,
                        count, n, k,
                        a_stride, c_stride
                    );
                }
            }
            return;
        }
        // Ruy path: ARM always, x86 for thin matrices
#pragma omp parallel for num_threads(g_n_threads) schedule(static)
        for (int t = 0; t < static_cast<int>(g_n_threads); ++t) {
            const size_t row_start = t * m / g_n_threads;
            const size_t row_end = (t + 1) * m / g_n_threads;
            const size_t local_rows = row_end - row_start;
            if (local_rows == 0) continue;

            thread_local ruy::Context ctx;
            ctx.set_max_num_threads(1);

            ruy::Matrix<std::uint8_t> lhs;
            lhs.mutable_layout()->set_rows(local_rows);
            lhs.mutable_layout()->set_cols(k);
            lhs.mutable_layout()->set_order(ruy::Order::kRowMajor);
            lhs.mutable_layout()->set_stride(a_stride);
            lhs.set_data(a + row_start * a_stride);

            ruy::Matrix<std::uint8_t> rhs;
            rhs.mutable_layout()->set_rows(k);
            rhs.mutable_layout()->set_cols(n);
            rhs.mutable_layout()->set_order(ruy::Order::kColMajor);
            rhs.mutable_layout()->set_stride(b_stride);
            rhs.set_data(b);

            ruy::Matrix<std::int32_t> dst;
            dst.mutable_layout()->set_rows(local_rows);
            dst.mutable_layout()->set_cols(n);
            dst.mutable_layout()->set_order(ruy::Order::kRowMajor);
            dst.mutable_layout()->set_stride(n);
            dst.set_data(reinterpret_cast<std::int32_t*>(out + row_start * n));

            ruy::MulParams<std::int32_t, std::int32_t> mul_params;
            ruy::Mul(lhs, rhs, mul_params, &ctx, &dst);
        }
    }

    void Fit(const float* embeddings, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("fitting");
        const size_t total_elements = n * d;
        params = ComputeQuantizationParams(embeddings, total_elements);
        tmp_dots_buf.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        centroids_nk_packed_buf.resize(nk_dots_packed_size_u8(Y_BATCH_SIZE, d));
        fitted = true;
    }

    static ScalarQuantizationParams ComputeQuantizationParams(
        const float* embeddings,
        const size_t total_elements
    ) {
        float global_min = std::numeric_limits<float>::max();
        float global_max = std::numeric_limits<float>::lowest();

#pragma omp parallel for reduction(min : global_min) reduction(max : global_max)                   \
    num_threads(g_n_threads)
        for (size_t i = 0; i < total_elements; ++i) {
            global_min = std::min(global_min, embeddings[i]);
            global_max = std::max(global_max, embeddings[i]);
        }

        const float range = global_max - global_min;
        const float scale = (range > 0) ? static_cast<float>(MAX_VALUE) / range : 1.0f;
        return {global_min, scale, 1.0f / scale};
    }

    void Encode(
        const float* embeddings,
        quantized_t* output_quantized_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("encoding");
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float quantization_scale = params.quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const float* embedding = embeddings + row * d;
            quantized_t* output_quantized_embedding = output_quantized_embeddings + row * d;
            for (size_t i = 0; i < d; ++i) {
                const int rounded = static_cast<int>(
                    std::round((embedding[i] - quantization_base) * quantization_scale)
                );
                if (SKM_UNLIKELY(rounded > MAX_VALUE)) {
                    output_quantized_embedding[i] = MAX_VALUE;
                } else if (SKM_UNLIKELY(rounded < 0)) {
                    output_quantized_embedding[i] = 0;
                } else {
                    output_quantized_embedding[i] = static_cast<uint8_t>(rounded);
                }
            }
        }
    }

    void Decode(
        const quantized_t* quantized_embeddings,
        float* output_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("decoding");
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float inv_quantization_scale = params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const quantized_t* quantized_embedding = quantized_embeddings + row * d;
            float* output_embedding = output_embeddings + row * d;
            for (size_t i = 0; i < d; ++i) {
                output_embedding[i] =
                    static_cast<float>(quantized_embedding[i]) * inv_quantization_scale +
                    quantization_base;
            }
        }
    }

    /**
     * @brief Compute float L2 squared norms of quantized vectors.
     *
     * Since base cancels in L2 distance:
     *   norm[i] = inv_scale² * Σ q[i][dim]²
     */
    void ComputeNorms(
        const quantized_t* quantized_embeddings, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted);
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const quantized_t* row = quantized_embeddings + i * d;
            uint32_t sum_sq = 0;
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                uint32_t v = row[j];
                sum_sq += v * v;
            }
            out_norms[i] = inv_scale_sq * static_cast<float>(sum_sq);
        }
    }

    /**
     * @brief Find top-1 nearest neighbor using ruy u8 dot product GEMM.
     *
     * Computes dot products via ruy, then converts to L2² using pre-computed
     * norms: L2²(x,y) = ||x||² + ||y||² - 2·dot(x,y).
     */
    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        SKM_PROFILE_SCOPE("search");
        SKM_PROFILE_SCOPE("search/1st_blas");
        assert(fitted);
        (void)x_float;
        (void)y_float;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;
        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                MatrixMultiplication(
                    x + i * d, y + j * d,
                    tmp_dots_buf.data(),
                    batch_n_x, batch_n_y, d, d, d
                );

                // L2²(x,y) = ||x||² + ||y||² - 2·inv_scale²·dot(x,y)
                // TODO(@lkuffo, low): I believe we can just avoid the inv_scale term
                using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<MatrixR> dists_matrix(tmp_buf, batch_n_x, batch_n_y);
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const size_t idx = i + r;
                    const uint32_t* dots_row = tmp_dots_buf.data() + r * batch_n_y;
                    float* dists_row = tmp_buf + r * batch_n_y;
                    const float nx = norms_x[idx];

                    SKM_VECTORIZE_LOOP
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        dists_row[c] = nx + norms_y[j + c]
                            - 2.0f * inv_scale_sq * static_cast<float>(dots_row[c]);
                    }

                    uint32_t knn_idx;
                    float batch_top_1 = dists_matrix.row(r).minCoeff(&knn_idx);
                    if (batch_top_1 < out_distances[idx]) {
                        out_distances[idx] = std::max(0.0f, batch_top_1);
                        out_knn[idx] = static_cast<uint32_t>(j + knn_idx);
                    }
                }
            }
        }
    }

    void CacheDataPartialNorms(
        const quantized_t* data, size_t n, size_t d, uint32_t partial_d
    ) override {
        cached_data_partial_norms.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = data + idx * d;
            SKM_VECTORIZE_LOOP
            for (size_t dim = 0; dim < partial_d; ++dim) {
                uint32_t v = row[dim];
                sum += v * v;
            }
            cached_data_partial_norms[idx] = sum;
        }
    }

    void CacheCentroidPartialNorms(
        const quantized_t* centroids, size_t n, size_t d, uint32_t partial_d
    ) override {
        cached_centroid_partial_norms.resize(n);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t idx = 0; idx < n; ++idx) {
            uint32_t sum = 0;
            const quantized_t* row = centroids + idx * d;
            SKM_VECTORIZE_LOOP
            for (size_t dim = 0; dim < partial_d; ++dim) {
                uint32_t v = row[dim];
                sum += v * v;
            }
            cached_centroid_partial_norms[idx] = sum;
        }
    }

    /**
     * @brief Find top-1 nearest neighbor with PDX pruning for u8.
     *
     * Hybrid approach that computes partial distances (first partial_d dimensions)
     * via GEMM, then uses ADSampling+PDX pruning to skip full distance computation
     * for unlikely candidates.
     * Final distances are float (converted inside PDXearch::SetBestCandidate).
     * Partial norms must be cached via CacheDataPartialNorms / CacheCentroidPartialNorms.
     */
    void FindNearestNeighborWithPruning(
        const quantized_t* x,
        const quantized_t* y,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        PDXLayout<Quantization::u8, DistanceFunction::l2>& pdx_centroids,
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) const override {
        SKM_PROFILE_SCOPE("search");
        (void) x_float;
        (void) y_float;

        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;

        pdx_centroids.index->quantization_scale_squared =
            params.quantization_scale * params.quantization_scale;
        pdx_centroids.index->inverse_scale_factor_squared = inv_scale_sq;

        // Buffers for batched dot products (uint32_t)
        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("search/blas");
                    MatrixMultiplication(
                        x + i * d, y + j * d,
                        tmp_dots_buf.data(),
                        batch_n_x, batch_n_y, partial_d, d, d
                    );
                }

                {
                    SKM_PROFILE_SCOPE("search/pdx");
                    // Convert dots to L2² and run PDXearch per query vector
#if defined(__clang__)
#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
#else
#pragma omp parallel for num_threads(g_n_threads)
#endif
                    for (size_t r = 0; r < batch_n_x; ++r) {
                        const size_t i_idx = i + r;

                        // Norms: convert dot products to squared L2 distances
                        const uint32_t norm_x_i = cached_data_partial_norms[i_idx];
                        uint32_t* partial_distances_p = tmp_dots_buf.data() + r * batch_n_y;
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            partial_distances_p[c] =
                                norm_x_i + cached_centroid_partial_norms[j + c] - 2 * partial_distances_p[c];
                        }

                        // PDX pruned search per vector
                        auto data_p = x + (i_idx * d);
                        const auto prev_assignment = out_knn[i_idx];
                        float dist_to_prev_centroid;
                        if (j == 0) {
                            uint32_t u8_dist = u8_computer::Horizontal(
                                data_p, y + prev_assignment * d, d
                            );
                            dist_to_prev_centroid = static_cast<float>(u8_dist) * inv_scale_sq;
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        // PDXearch with uint32_t partial distances
                        size_t local_not_pruned = 0;
                        auto assignment =
                            pdx_centroids.searcher
                                ->Top1PartialSearchWithThresholdAndPartialDistances(
                                    data_p,
                                    dist_to_prev_centroid,
                                    prev_assignment,
                                    partial_distances_p,
                                    partial_d,
                                    j / VECTOR_CHUNK_SIZE,
                                    (j + Y_BATCH_SIZE) / VECTOR_CHUNK_SIZE,
                                    local_not_pruned
                                );
                        out_not_pruned_counts[i_idx] += local_not_pruned;
                        out_knn[i_idx] = assignment.index;
                        out_distances[i_idx] = assignment.distance;
                    }
                }
            }
        }
    }

    void AverageCentroids(
        const uint32_t* accumulators,
        const uint32_t* cluster_sizes,
        quantized_t* out,
        size_t n_clusters,
        size_t d
    ) const override {
        assert(fitted);
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_clusters; ++i) {
            if (cluster_sizes[i] == 0) continue;
            const uint32_t* acc = accumulators + i * d;
            quantized_t* row = out + i * d;
            const uint32_t half = cluster_sizes[i] / 2;
            const float inv_size = 1.0f / static_cast<float>(cluster_sizes[i]);
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                row[j] = static_cast<uint8_t>(static_cast<float>(acc[j] + half) * inv_size);
            }
        }
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return true; }
    const ScalarQuantizationParams& GetParams() const { return params; }

  private:
    ScalarQuantizationParams params{};
    bool fitted = false;
    bool has_amx = false;
    std::vector<uint32_t> cached_data_partial_norms;
    std::vector<uint32_t> cached_centroid_partial_norms;
    mutable std::vector<uint32_t> tmp_dots_buf;
    mutable std::vector<char> centroids_nk_packed_buf;
};

} // namespace skmeans
