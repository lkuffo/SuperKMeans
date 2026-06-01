#pragma once

#include "superkmeans/common.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <Eigen/Dense>
#include <cassert>
#include <cstring>
#include <memory>
#include <omp.h>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace skmeans {

/**
 * @brief HNSW-over-SQ8 "quantizer" hack (FAISS backend).
 *
 * Mirrors HNSWQuantizer but uses faiss::IndexHNSWSQ as storage. Centroids are
 * SQ8-encoded by FAISS on add(); distance during search is asymmetric
 * (float query → SQ8 centroid) computed by FAISS's LUT-based kernels.
 *
 * Per-iteration the SQ ranges are re-fit on the centroids. With ~k_clusters
 * samples this is cheap and adapts to centroid drift.
 *
 * Supports the same warm-start trick as HNSWQuantizer via search_level_0:
 * after the first call, out_knn already holds the previous iteration's
 * assignments and is used as per-query entry points.
 */
class HNSWSQ8Quantizer : public IQuantizer<Quantization::f32> {
  public:
    using quantized_t = IQuantizer::quantized_t; // float (interface layer)
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;

    HNSWSQ8Quantizer(int hnsw_M = 32, int ef_construction = 40, int ef_search = 16,
                     bool use_warm_start = false)
        : hnsw_M(hnsw_M), ef_construction(ef_construction), ef_search(ef_search),
          use_warm_start(use_warm_start) {}

    void Fit(const float* /*data*/, size_t /*n*/, size_t d) override {
        dim = d;
        fitted = true;
        first_call_done = false;
    }

    void Encode(const float* in, float* out, size_t n, size_t d) const override {
        assert(fitted);
        if (in != out) {
            std::memcpy(out, in, n * d * sizeof(float));
        }
    }

    void Decode(const float* in, float* out, size_t n, size_t d) const override {
        assert(fitted);
        if (in != out) {
            std::memcpy(out, in, n * d * sizeof(float));
        }
    }

    void ComputeNorms(
        const float* data, size_t n, size_t d, float* out_norms
    ) const override {
        assert(fitted);
        Eigen::Map<const MatrixR> e_data(data, n, d);
        Eigen::Map<VectorR> e_norms(out_norms, n);
        e_norms.noalias() = e_data.rowwise().squaredNorm();
    }

    /**
     * @brief Build a fresh IndexHNSWSQ over the centroids and query top-1 per data point.
     */
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

        faiss::IndexHNSWSQ index(
            static_cast<int>(d),
            faiss::ScalarQuantizer::QT_8bit,
            hnsw_M,
            faiss::METRIC_L2
        );
        index.hnsw.efConstruction = ef_construction;
        index.hnsw.efSearch = ef_search;
        {
            SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor/construction");
            // train() fits SQ ranges on the centroids, add() encodes and inserts.
            index.train(static_cast<faiss::idx_t>(n_y), y);
            index.add(static_cast<faiss::idx_t>(n_y), y);
        }

        std::unique_ptr<faiss::idx_t[]> labels(new faiss::idx_t[n_x]);
        if (use_warm_start && first_call_done) {
            std::vector<faiss::HNSW::storage_idx_t> entry_points(n_x);
            std::vector<float> entry_dists(n_x);
            {
                SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor/entry_points");
#pragma omp parallel for if (g_n_threads > 1) num_threads(g_n_threads)
                for (size_t i = 0; i < n_x; ++i) {
                    uint32_t c = out_knn[i];
                    if (c >= n_y) c = 0;
                    entry_points[i] = static_cast<faiss::HNSW::storage_idx_t>(c);
                    const float* xi = x + i * d;
                    const float* yc = y + c * d;
                    float dist = 0.0f;
                    for (size_t j = 0; j < d; ++j) {
                        float diff = xi[j] - yc[j];
                        dist += diff * diff;
                    }
                    entry_dists[i] = dist;
                }
            }
            {
                SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor/search_level_0");
                index.search_level_0(
                    static_cast<faiss::idx_t>(n_x),
                    x,
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
            SKM_PROFILE_SCOPE("HNSW_SQ8::FindNearestNeighbor/search");
            index.search(
                static_cast<faiss::idx_t>(n_x),
                x,
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

    void UpdateCentroids(
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* centroid_accumulators,
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
                    const float* vec = encoded_data + i * d;
                    float* acc = centroid_accumulators + ci * d;
                    cluster_sizes[ci] += 1;
                    SKM_VECTORIZE_LOOP
                    for (size_t j = 0; j < d; ++j) {
                        acc[j] += vec[j];
                    }
                }
            }
        }
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return false; }
    size_t CodeSize(size_t d) const override { return d; }

  private:
    bool fitted = false;
    size_t dim = 0;
    int hnsw_M;
    int ef_construction;
    int ef_search;
    bool use_warm_start;
    mutable bool first_call_done = false;
};

} // namespace skmeans
