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
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace skmeans {

/**
 * @brief HNSW-over-SQ8 quantizer backed by FAISS, with a fair-comparison twist.
 *
 * FAISS's IndexHNSWSQ is inherently asymmetric: queries arrive as float and
 * are compared against SQ8-encoded storage. To put it on equal footing with
 * the symmetric u8↔u8 USearch path (HNSWSQ8Quantizer), this quantizer routes
 * queries through an f32 → SQ8 → f32 round-trip before handing them to FAISS.
 * That preserves the same information loss the symmetric path imposes, while
 * keeping FAISS's asymmetric search kernel.
 *
 * Operates at the u8 layer so the framework caches the encoded data once
 * (sq_common.h's ScalarQuantizationParams). Both data and centroids are
 * decoded from their cached u8 representation on every iteration before
 * being passed to FAISS. The decoded-data buffer is cached internally
 * (it doesn't change between iterations); decoded centroids re-derive each call.
 */
class HNSWFaissSQ8Quantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t; // uint8_t

    static constexpr uint8_t MAX_VALUE = 255;

    HNSWFaissSQ8Quantizer(int hnsw_M = 32, int ef_construction = 40, int ef_search = 16,
                          bool use_warm_start = false)
        : hnsw_M(hnsw_M), ef_construction(ef_construction), ef_search(ef_search),
          use_warm_start(use_warm_start) {}

    void Fit(const float* embeddings, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::Fit");
        dim = d;
        params = ComputeScalarQuantizationParams(
            embeddings, n * d, static_cast<float>(MAX_VALUE)
        );
        fitted = true;
        cached_data_ptr = nullptr;
        decoded_data.clear();
        first_call_done = false;
    }

    void Encode(
        const float* embeddings,
        quantized_t* output_quantized_embeddings,
        size_t n,
        size_t d
    ) const override {
        SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::Encode");
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
        SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::Decode");
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
        SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::FindNearestNeighbor");
        assert(fitted);

        // Decode centroids on every call (they drift each iteration).
        std::vector<float> y_decoded(n_y * d);
        Decode(y, y_decoded.data(), n_y, d);

        // Decode data once and cache. The data pointer is stable across iterations
        // (framework allocates it once in Train), so a pointer-keyed cache is safe.
        if (cached_data_ptr != x || decoded_data.size() != n_x * d) {
            decoded_data.assign(n_x * d, 0.0f);
            Decode(x, decoded_data.data(), n_x, d);
            cached_data_ptr = x;
        }

        faiss::IndexHNSWSQ index(
            static_cast<int>(d),
            faiss::ScalarQuantizer::QT_8bit,
            hnsw_M,
            faiss::METRIC_L2
        );
        index.hnsw.efConstruction = ef_construction;
        index.hnsw.efSearch = ef_search;
        {
            SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::FindNearestNeighbor/construction");
            // FAISS will (re)quantize these decoded floats to SQ8 internally.
            index.train(static_cast<faiss::idx_t>(n_y), y_decoded.data());
            index.add(static_cast<faiss::idx_t>(n_y), y_decoded.data());
        }

        std::unique_ptr<faiss::idx_t[]> labels(new faiss::idx_t[n_x]);
        if (use_warm_start && first_call_done) {
            // Warm-start path: out_knn carries the previous iteration's assignments.
            // Use them as per-query level-0 entry points; distances to entries are
            // computed on the decoded floats (same domain FAISS searches in).
            std::vector<faiss::HNSW::storage_idx_t> entry_points(n_x);
            std::vector<float> entry_dists(n_x);
            {
                SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::FindNearestNeighbor/entry_points");
#pragma omp parallel for if (g_n_threads > 1) num_threads(g_n_threads)
                for (size_t i = 0; i < n_x; ++i) {
                    uint32_t c = out_knn[i];
                    if (c >= n_y) c = 0;
                    entry_points[i] = static_cast<faiss::HNSW::storage_idx_t>(c);
                    const float* xi = decoded_data.data() + i * d;
                    const float* yc = y_decoded.data() + c * d;
                    float dist = 0.0f;
                    for (size_t j = 0; j < d; ++j) {
                        float diff = xi[j] - yc[j];
                        dist += diff * diff;
                    }
                    entry_dists[i] = dist;
                }
            }
            {
                SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::FindNearestNeighbor/search_level_0");
                index.search_level_0(
                    static_cast<faiss::idx_t>(n_x),
                    decoded_data.data(),
                    1,
                    entry_points.data(),
                    entry_dists.data(),
                    out_distances,
                    labels.get(),
                    /*nprobe=*/1,
                    /*search_type=*/1
                );
            }
        } else {
            SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::FindNearestNeighbor/search");
            index.search(
                static_cast<faiss::idx_t>(n_x),
                decoded_data.data(),
                1,
                out_distances,
                labels.get()
            );
        }
        for (size_t i = 0; i < n_x; ++i) {
            out_knn[i] = static_cast<uint32_t>(labels[i]);
        }
        first_call_done = true;
    }

    void ResetCentroidAccumulators(size_t n_clusters, size_t d) override {
        const size_t total = n_clusters * d;
        if (centroid_accumulators.size() < total) centroid_accumulators.resize(total);
        std::fill_n(centroid_accumulators.data(), total, 0u);
    }

    void UpdateCentroids(
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* /*centroid_accumulators_float*/,
        uint32_t* cluster_sizes,
        size_t n, size_t n_clusters, size_t d,
        uint32_t n_threads
    ) const override {
        SKM_PROFILE_SCOPE("HNSW_FAISS_SQ8::UpdateCentroids");
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

    void InvalidateCaches() override {
        cached_data_ptr = nullptr;
        decoded_data.clear();
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
    bool use_warm_start;
    mutable bool first_call_done = false;
    mutable std::vector<uint32_t> centroid_accumulators;
    // Pointer-keyed cache for the decoded-float data (built once per Train).
    mutable const quantized_t* cached_data_ptr = nullptr;
    mutable std::vector<float> decoded_data;
};

} // namespace skmeans
