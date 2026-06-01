#pragma once

#include "superkmeans/common.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/sq_common.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <vector>

#include <usearch/index_dense.hpp>

namespace skmeans {

/**
 * @brief HNSW-over-SQ8 quantizer backed by USearch (symmetric u8↔u8 distance).
 *
 * Operates at the u8 layer so the SuperKMeans framework caches the encoded
 * data once during Train() and reuses it across iterations (no re-quantization
 * of the data per iteration). Centroids are re-encoded each iteration by the
 * framework (they change every iteration).
 *
 * The float→u8 encoding reuses sq_common.h's ScalarQuantizationParams:
 *   q[i] = clamp(round((v[i] - base) * scale), 0, 255)
 * For L2 distance the base cancels:
 *   ||x-y||²_float = inv_scale² · ||x_q - y_q||²_u8
 * so we recover float distances from usearch's u8 L2² by multiplying by inv_scale².
 *
 * USearch's public search API does not expose per-query entry points, so
 * warm-start is not supported on this backend.
 */
class HNSWSQ8Quantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t; // uint8_t
    using usearch_index_t = unum::usearch::index_dense_t;
    using usearch_metric_t = unum::usearch::metric_punned_t;

    static constexpr uint8_t MAX_VALUE = 255;

    HNSWSQ8Quantizer(int hnsw_M = 32, int ef_construction = 40, int ef_search = 16)
        : hnsw_M(hnsw_M), ef_construction(ef_construction), ef_search(ef_search) {}

    void Fit(const float* embeddings, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("HNSW_SQ8::Fit");
        dim = d;
        params = ComputeScalarQuantizationParams(
            embeddings, n * d, static_cast<float>(MAX_VALUE)
        );
        fitted = true;
    }

    void Encode(
        const float* embeddings,
        quantized_t* output_quantized_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("HNSW_SQ8::Encode");
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float quantization_scale = params.quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const float* embedding = embeddings + row * d;
            quantized_t* out_row = output_quantized_embeddings + row * d;
            for (size_t i = 0; i < d; ++i) {
                const int rounded = static_cast<int>(
                    std::round((embedding[i] - quantization_base) * quantization_scale)
                );
                if (SKM_UNLIKELY(rounded > MAX_VALUE)) {
                    out_row[i] = MAX_VALUE;
                } else if (SKM_UNLIKELY(rounded < 0)) {
                    out_row[i] = 0;
                } else {
                    out_row[i] = static_cast<uint8_t>(rounded);
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
        SKM_PROFILE_SCOPE("HNSW_SQ8::Decode");
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float inv_quantization_scale = params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t row = 0; row < n; ++row) {
            const quantized_t* in_row = quantized_embeddings + row * d;
            float* out_row = output_embeddings + row * d;
            for (size_t i = 0; i < d; ++i) {
                out_row[i] = static_cast<float>(in_row[i]) * inv_quantization_scale
                             + quantization_base;
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

    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        const float* /*norms_x*/,
        const float* /*norms_y*/,
        uint32_t* out_knn,
        float* out_distances,
        float* /*tmp_buf*/
    ) const override {
        SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor");
        assert(fitted);

        // Build a fresh USearch index over the (already-encoded) centroids.
        usearch_metric_t metric{
            d,
            unum::usearch::metric_kind_t::l2sq_k,
            unum::usearch::scalar_kind_t::u8_k
        };
        unum::usearch::index_dense_config_t cfg(
            static_cast<size_t>(hnsw_M),
            static_cast<size_t>(ef_construction),
            static_cast<size_t>(ef_search)
        );
        unum::usearch::index_limits_t limits(n_y, g_n_threads);
        auto state = usearch_index_t::make(metric, cfg, /*free_key=*/static_cast<unum::usearch::default_key_t>(-1), limits);
        if (!state) {
            throw std::runtime_error(
                std::string("USearch index init failed: ") + state.error.what()
            );
        }
        auto& index = state.index;

        {
            SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor/construction");
            // Sequential add: parallel add to USearch produces unreachable nodes
            // (~2% of n_y end up disconnected from the entry point's graph),
            // which causes many empty clusters and runaway split counts in k-means.
            for (size_t i = 0; i < n_y; ++i) {
                index.add(
                    static_cast<unum::usearch::default_key_t>(i),
                    y + i * d,
                    0
                );
            }
        }

        const float inv_scale_sq =
            params.inv_quantization_scale * params.inv_quantization_scale;
        {
            SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor/search");
#pragma omp parallel for if (g_n_threads > 1) num_threads(g_n_threads)
            for (size_t i = 0; i < n_x; ++i) {
                auto result = index.search(
                    x + i * d, 1,
                    static_cast<size_t>(omp_get_thread_num())
                );
                out_knn[i] = static_cast<uint32_t>(result[0].member.key);
                // USearch returns L2² in u8 space; convert back to float space.
                out_distances[i] = inv_scale_sq * static_cast<float>(result[0].distance);
            }
        }
    }

    void ResetCentroidAccumulators(size_t n_clusters, size_t d) override {
        const size_t total = n_clusters * d;
        if (centroid_accumulators.size() < total) centroid_accumulators.resize(total);
        std::fill_n(centroid_accumulators.data(), total, 0u);
    }

    /**
     * @brief Update centroids via integer u8 accumulation (same scheme as SQ8Quantizer).
     */
    void UpdateCentroids(
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* /*centroid_accumulators_float*/,
        uint32_t* cluster_sizes,
        size_t n, size_t n_clusters, size_t d,
        uint32_t n_threads
    ) const override {
        SKM_PROFILE_SCOPE("HNSW_SQ8::UpdateCentroids");
        assert(fitted);
#pragma omp parallel if (n_threads > 1) num_threads(n_threads)
        {
            uint32_t nt = n_threads;
            uint32_t rank = omp_get_thread_num();
            size_t c0 = (n_clusters * rank) / nt;
            size_t c1 = (n_clusters * (rank + 1)) / nt;
            for (size_t i = 0; i < n; ++i) {
                uint32_t ci = assignments[i];
                if (ci >= c0 && ci < c1) {
                    cluster_sizes[ci] += 1;
                    const auto* vec = encoded_data + i * d;
                    auto* acc = centroid_accumulators.data() + ci * d;
                    SKM_VECTORIZE_LOOP
                    for (size_t j = 0; j < d; ++j) {
                        acc[j] += vec[j];
                    }
                }
            }
        }
    }

    void FinalizeCentroids(
        float* centroids,
        const uint32_t* cluster_sizes,
        size_t n_clusters, size_t d
    ) const override {
        assert(fitted);
        const float quantization_base = params.quantization_base;
        const float inv_quantization_scale = params.inv_quantization_scale;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_clusters; ++i) {
            if (cluster_sizes[i] == 0) continue;
            const uint32_t* acc = centroid_accumulators.data() + i * d;
            float* row = centroids + i * d;
            const uint32_t half = cluster_sizes[i] / 2;
            const float inv_size = 1.0f / static_cast<float>(cluster_sizes[i]);
            SKM_VECTORIZE_LOOP
            for (size_t j = 0; j < d; ++j) {
                uint8_t qval = static_cast<uint8_t>(static_cast<float>(acc[j] + half) * inv_size);
                row[j] = static_cast<float>(qval) * inv_quantization_scale + quantization_base;
            }
        }
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return false; }
    size_t CodeSize(size_t d) const override { return d; }

  private:
    ScalarQuantizationParams params{};
    bool fitted = false;
    size_t dim = 0;
    int hnsw_M;
    int ef_construction;
    int ef_search;
    mutable std::vector<uint32_t> centroid_accumulators;
};

} // namespace skmeans
