#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/sq8.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <numkong/numkong.h>

namespace skmeans {

/**
 * @brief 8-bit scalar quantizer with NumKong GEMM backend (for A/B comparison with ruy).
 *
 * Identical to SQ8Quantizer except uses NumKong nk_dots_packed_u8 for the
 * matrix multiply in both FindNearestNeighbor and FindNearestNeighborWithPruning.
 */
class SQ8NKQuantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;

    static constexpr uint8_t MAX_VALUE = 255;

    void Fit(const float* embeddings, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("fitting");
        const size_t total_elements = n * d;
        params = SQ8Quantizer::ComputeQuantizationParams(embeddings, total_elements);
        pruning_dots_buf.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        packed_buf.resize(nk_dots_packed_size_u8(Y_BATCH_SIZE, d));
        fitted = true;
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
     * @brief Find top-1 nearest neighbor using NumKong u8 dot product GEMM.
     *
     * Computes dot products via nk_dots_packed_u8, then converts to L2² using
     * pre-computed norms: L2²(x,y) = ||x||² + ||y||² - 2·inv_scale²·dot(x,y).
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
        std::fill_n(out_knn, n_x, 0u);

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                // Pack Y batch and compute dot products via NumKong
                {
                    const size_t pack_size = nk_dots_packed_size_u8(batch_n_y, d);
                    if (pack_size > packed_buf.size()) packed_buf.resize(pack_size);
                    nk_dots_pack_u8(y + j * d, batch_n_y, d, d, packed_buf.data());

                    const size_t c_stride = batch_n_y * sizeof(uint32_t);

#pragma omp parallel num_threads(g_n_threads)
                    {
                        nk_configure_thread(nk_capabilities());
                        int tid = omp_get_thread_num();
                        int nt = omp_get_num_threads();
                        size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                        size_t start = tid * rows_per_t;
                        size_t count = std::min(rows_per_t, batch_n_x - start);
                        if (start < batch_n_x && count > 0) {
                            nk_dots_packed_u8(
                                x + (i + start) * d,
                                packed_buf.data(),
                                pruning_dots_buf.data() + start * batch_n_y,
                                count,
                                batch_n_y,
                                d,
                                d,
                                c_stride
                            );
                        }
                    }
                }

                // Convert dots to L2² and find nearest neighbor per row
                using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                Eigen::Map<MatrixR> dists_matrix(tmp_buf, batch_n_x, batch_n_y);
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const size_t idx = i + r;
                    const uint32_t* dots_row = pruning_dots_buf.data() + r * batch_n_y;
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

    size_t DefaultRerankK() const override { return 0; }

    void FindNearestNeighborWithReranking(
        const quantized_t* x_quantized,
        const quantized_t* y_quantized,
        const float* x_float,
        const float* y_float,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* norms_x,
        const float* norms_y,
        size_t rerank_k,
        uint32_t* out_knn,
        float* out_distances,
        float* tmp_buf
    ) const override {
        SKM_PROFILE_SCOPE("search");
        SKM_PROFILE_SCOPE("search/rerank");
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;
    }

    bool IsFitted() const override { return fitted; }

    bool SupportsPruning() const override { return true; }

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
     * @brief Find top-1 nearest neighbor with PDX pruning using NumKong u8 dots.
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
        assert(fitted);
        (void) x_float;
        (void) y_float;
        assert(!cached_data_partial_norms.empty() && "CacheDataPartialNorms must be called first");
        assert(
            !cached_centroid_partial_norms.empty() &&
            "CacheCentroidPartialNorms must be called first"
        );

        using u8_computer = DistanceComputer<DistanceFunction::l2, Quantization::u8>;
        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;

        pdx_centroids.index->quantization_scale_squared =
            params.quantization_scale * params.quantization_scale;
        pdx_centroids.index->inverse_scale_factor_squared = inv_scale_sq;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("search/blas");
                    // Pack y for partial_d dimensions (stride = d for full vectors)
                    const size_t pack_size = nk_dots_packed_size_u8(batch_n_y, partial_d);
                    if (pack_size > packed_buf.size()) packed_buf.resize(pack_size);
                    nk_dots_pack_u8(y + j * d, batch_n_y, partial_d, d, packed_buf.data());

                    const size_t c_stride = batch_n_y * sizeof(uint32_t);

#pragma omp parallel num_threads(g_n_threads)
                    {
                        nk_configure_thread(nk_capabilities());
                        int tid = omp_get_thread_num();
                        int nt = omp_get_num_threads();
                        size_t rows_per_t = (batch_n_x + nt - 1) / nt;
                        size_t start = tid * rows_per_t;
                        size_t count = std::min(rows_per_t, batch_n_x - start);
                        if (start < batch_n_x && count > 0) {
                            nk_dots_packed_u8(
                                x + (i + start) * d,
                                packed_buf.data(),
                                pruning_dots_buf.data() + start * batch_n_y,
                                count,
                                batch_n_y,
                                partial_d,
                                d,
                                c_stride
                            );
                        }
                    }
                }

                {
                    SKM_PROFILE_SCOPE("search/pdx");
#if defined(__clang__)
#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
#else
#pragma omp parallel for num_threads(g_n_threads)
#endif
                    for (size_t r = 0; r < batch_n_x; ++r) {
                        const size_t i_idx = i + r;
                        uint32_t* dots_row = pruning_dots_buf.data() + r * batch_n_y;

                        const uint32_t nx = cached_data_partial_norms[i_idx];
                        SKM_VECTORIZE_LOOP
                        for (size_t c = 0; c < batch_n_y; ++c) {
                            dots_row[c] =
                                nx + cached_centroid_partial_norms[j + c] - 2 * dots_row[c];
                        }

                        const auto prev_assignment = out_knn[i_idx];
                        float dist_to_prev_centroid;
                        if (j == 0) {
                            uint32_t u8_dist = u8_computer::Horizontal(
                                x + i_idx * d, y + prev_assignment * d, d
                            );
                            dist_to_prev_centroid = static_cast<float>(u8_dist) * inv_scale_sq;
                        } else {
                            dist_to_prev_centroid = out_distances[i_idx];
                        }

                        size_t local_not_pruned = 0;
                        auto assignment =
                            pdx_centroids.searcher
                                ->Top1PartialSearchWithThresholdAndPartialDistances(
                                    x + i_idx * d,
                                    dist_to_prev_centroid,
                                    prev_assignment,
                                    dots_row,
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

    const ScalarQuantizationParams& GetParams() const { return params; }

  private:
    ScalarQuantizationParams params{};
    bool fitted = false;
    std::vector<uint32_t> cached_data_partial_norms;
    std::vector<uint32_t> cached_centroid_partial_norms;
    mutable std::vector<uint32_t> pruning_dots_buf;
    mutable std::vector<char> packed_buf;
};

} // namespace skmeans
