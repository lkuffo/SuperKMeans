#pragma once

#include "superkmeans/common.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/utils/quantize_lut.h>

namespace skmeans {

/**
 * @brief 4-bit Product Quantizer (Ks=16) with FAISS FastScan SIMD distance.
 *
 * Same subspace decomposition as PQ8 but with only 16 codewords per subspace
 * (4 bits). The key advantage is FastScan SIMD: vpshufb/vtbl can perform 16
 * parallel table lookups in a single instruction, enabling distance computation
 * for 32 database vectors at a time.
 *
 * Typical config: larger M to compensate for fewer codewords per subspace.
 * E.g. d=768: M=96 dsub=8 (vs M=16 dsub=48 for 8-bit PQ).
 *
 * SDC table is tiny: M × 16 × 16 × 4 bytes = M × 1024 bytes (fits in L1).
 *
 * Centroid updates use the same BLAS-accelerated sparse voting as PQ8.
 */
class PQ4Quantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;

    static constexpr size_t Ks = 16;
    static constexpr size_t BBS = 32; // FastScan block size
    static constexpr size_t NQ_BATCH = 4; // queries per accumulate_to_mem call

    explicit PQ4Quantizer(size_t M) : M_(M) {
        assert(M_ % 2 == 0 && "M must be even for 4-bit PQ");
    }

    void Fit(const float* data, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("fitting");
        assert(d % M_ == 0 && "d must be divisible by M");
        d_ = d;
        dsub_ = d / M_;
        code_size_ = M_ / 2; // 2 nibbles per byte
        nsq_ = M_; // M is already even

        faiss_pq_ = std::make_unique<faiss::ProductQuantizer>(d, M_, 4);
        faiss_pq_->verbose = false;
        faiss_pq_->train(n, data);
        faiss_pq_->compute_sdc_table();

        // Copy SDC table: layout [M × Ks × Ks]
        sdc_table_ = faiss_pq_->sdc_table;
        assert(sdc_table_.size() == M_ * Ks * Ks);

        fitted = true;
    }

    void Encode(const float* in, quantized_t* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("encoding");
        assert(fitted);
        assert(d == d_);
        faiss_pq_->compute_codes(in, out, n);
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("decoding");
        assert(fitted);
        assert(d == d_);
        faiss_pq_->decode(in, out, n);
    }

    size_t CodeSize(size_t /*d*/) const override { return code_size_; }

    void ComputeNorms(
        const quantized_t* /*data*/, size_t n, size_t /*d*/, float* out_norms
    ) const override {
        std::fill_n(out_norms, n, 0.0f);
    }

    /**
     * @brief Find top-1 nearest neighbor using FAISS FastScan SIMD.
     *
     * Centroids (y) are packed into FastScan block format. Data points (x)
     * are processed in batches of NQ_BATCH=4 queries per accumulate_to_mem
     * call so the centroid data is loaded once and reused across queries.
     *
     * Quantized LUTs and normalization factors for x are cached across
     * iterations (they depend only on data codes, not centroids).
     */
    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t /*d*/,
        const float* /*norms_x*/,
        const float* /*norms_y*/,
        uint32_t* out_knn,
        float* out_distances,
        float* /*tmp_buf*/
    ) const override {
        SKM_PROFILE_SCOPE("search");
        assert(fitted);

        // Cache quantized LUTs + norms if data changed
        EnsureLUTCache(x, n_x);

        // Pack centroids (y) into FastScan block format
        const size_t ntotal2 = ((n_y + BBS - 1) / BBS) * BBS;
        std::vector<uint8_t> packed_centroids(ntotal2 * nsq_ / 2, 0);
        faiss::pq4_pack_codes(y, n_y, M_, ntotal2, BBS, nsq_, packed_centroids.data());

        const size_t dim12 = Ks * nsq_;

#pragma omp parallel num_threads(g_n_threads)
        {
            // Per-thread buffers sized for NQ_BATCH queries
            std::vector<uint8_t> batch_quant_lut(NQ_BATCH * dim12);
            std::vector<uint8_t> packed_lut(NQ_BATCH * dim12);
            std::vector<uint16_t> accu(NQ_BATCH * ntotal2);

#pragma omp for schedule(dynamic, 16)
            for (size_t i_base = 0; i_base < n_x; i_base += NQ_BATCH) {
                const size_t actual_nq = std::min(NQ_BATCH, n_x - i_base);

                // Copy cached quantized LUTs for this batch of queries
                for (size_t q = 0; q < actual_nq; ++q) {
                    std::memcpy(
                        batch_quant_lut.data() + q * dim12,
                        cached_quant_luts_.data() + (i_base + q) * dim12,
                        dim12
                    );
                }

                // Pack LUTs for SIMD (interleaved across nq queries)
                faiss::pq4_pack_LUT(
                    static_cast<int>(actual_nq),
                    static_cast<int>(nsq_),
                    batch_quant_lut.data(),
                    packed_lut.data()
                );

                // Run SIMD accumulation for all queries in this batch
                faiss::accumulate_to_mem(
                    static_cast<int>(actual_nq), ntotal2, static_cast<int>(nsq_),
                    packed_centroids.data(), packed_lut.data(), accu.data()
                );

                // Find minimum distance centroid for each query
                for (size_t q = 0; q < actual_nq; ++q) {
                    const uint16_t* accu_q = accu.data() + q * ntotal2;
                    uint16_t best_accu = accu_q[0];
                    uint32_t best_j = 0;
                    for (size_t j = 1; j < n_y; ++j) {
                        if (accu_q[j] < best_accu) {
                            best_accu = accu_q[j];
                            best_j = static_cast<uint32_t>(j);
                        }
                    }
                    const size_t idx = i_base + q;
                    out_knn[idx] = best_j;
                    out_distances[idx] =
                        static_cast<float>(best_accu) / cached_norm_a_[idx] + cached_norm_b_[idx];
                }
            }
        }
    }

    size_t DefaultRerankK() const override { return 0; }

    void FindNearestNeighborWithReranking(
        const quantized_t* x_quantized,
        const quantized_t* y_quantized,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t /*d*/,
        const float* /*norms_x*/,
        const float* /*norms_y*/,
        size_t /*rerank_k*/,
        uint32_t* out_knn,
        float* out_distances,
        float* /*tmp_buf*/
    ) const override {
        FindNearestNeighbor(
            x_quantized, y_quantized,
            nullptr, nullptr,
            n_x, n_y, 0,
            nullptr, nullptr,
            out_knn, out_distances, nullptr
        );
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return false; }
    bool UsesSparseVoting() const override { return true; }

    /**
     * @brief Update centroids via sparse voting with 16×16 SDC tables.
     *
     * Same BLAS sgemm approach as PQ8 but with Ks=16:
     *   votes[n_clusters × 16] = freq[n_clusters × 16] × SDC_m[16 × 16]
     */
    void SparseVotingUpdate(
        const quantized_t* codes, const uint32_t* assignments,
        quantized_t* out_centroids, uint32_t* out_cluster_sizes,
        size_t n, size_t n_clusters, size_t /*d*/
    ) const override {
        SKM_PROFILE_SCOPE("sparse_voting");
        assert(fitted);

        // Count cluster sizes
        std::fill_n(out_cluster_sizes, n_clusters, 0u);
        for (size_t i = 0; i < n; ++i) {
            out_cluster_sizes[assignments[i]]++;
        }

        // Zero output centroids
        std::fill_n(out_centroids, n_clusters * code_size_, 0u);

        // Frequency matrix and votes per subspace
        std::vector<float> freq(n_clusters * Ks);
        std::vector<float> votes(n_clusters * Ks);

        for (size_t m = 0; m < M_; ++m) {
            // Build frequency histograms for subspace m
            std::fill(freq.begin(), freq.end(), 0.0f);
            for (size_t i = 0; i < n; ++i) {
                uint32_t cluster = assignments[i];
                uint8_t code = GetCode(codes + i * code_size_, m);
                freq[cluster * Ks + code] += 1.0f;
            }

            // BLAS sgemm: votes = freq × SDC_m
            // Row-major → Fortran column-major convention
            int blas_m = static_cast<int>(Ks);
            int blas_n = static_cast<int>(n_clusters);
            int blas_k = static_cast<int>(Ks);
            float alpha_val = 1.0f;
            float beta_val = 0.0f;
            int lda = static_cast<int>(Ks);
            int ldb = static_cast<int>(Ks);
            int ldc = static_cast<int>(Ks);

            sgemm_(
                "N", "N",
                &blas_m, &blas_n, &blas_k,
                &alpha_val,
                sdc_table_.data() + m * Ks * Ks, &lda,
                freq.data(), &ldb,
                &beta_val,
                votes.data(), &ldc
            );

            // Pick argmin per cluster and pack into nibbles
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t c = 0; c < n_clusters; ++c) {
                if (out_cluster_sizes[c] == 0) {
                    SetCode(out_centroids + c * code_size_, m, 0);
                    continue;
                }
                const float* row = votes.data() + c * Ks;
                size_t best_k = 0;
                float best_vote = row[0];
                for (size_t k = 1; k < Ks; ++k) {
                    if (row[k] < best_vote) {
                        best_vote = row[k];
                        best_k = k;
                    }
                }
                SetCode(out_centroids + c * code_size_, m, static_cast<uint8_t>(best_k));
            }
        }
    }

    void AverageCentroids(
        const uint32_t* /*accumulators*/,
        const uint32_t* /*cluster_sizes*/,
        quantized_t* /*out*/,
        size_t /*n_clusters*/,
        size_t /*d*/
    ) const override {
        // Not used — PQ4 uses SparseVotingUpdate instead
    }

  private:
    /// Extract 4-bit code for subspace m from a packed code vector.
    static uint8_t GetCode(const uint8_t* code, size_t m) {
        uint8_t byte = code[m / 2];
        return (m % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    }

    /// Set 4-bit code for subspace m in a packed code vector.
    static void SetCode(uint8_t* code, size_t m, uint8_t val) {
        size_t byte_idx = m / 2;
        if (m % 2 == 0) {
            code[byte_idx] = (code[byte_idx] & 0xF0) | (val & 0x0F);
        } else {
            code[byte_idx] = (code[byte_idx] & 0x0F) | ((val & 0x0F) << 4);
        }
    }

    /**
     * @brief Build and cache quantized LUTs + normalization factors for all data points.
     *
     * Each data point's SDC-based float LUT is quantized to uint8 via
     * round_uint8_per_column. The result depends only on x codes (not centroids),
     * so it's computed once and reused across k-means iterations.
     *
     * Memory: n_x × (M×16 + 8) bytes. For M=16, n=1M: ~17 MB.
     */
    void EnsureLUTCache(const quantized_t* x, size_t n_x) const {
        if (cached_x_ptr_ == x && cached_n_x_ == n_x) return;

        const size_t dim12 = Ks * nsq_;
        cached_quant_luts_.resize(n_x * dim12);
        cached_norm_a_.resize(n_x);
        cached_norm_b_.resize(n_x);

#pragma omp parallel num_threads(g_n_threads)
        {
            std::vector<float> float_lut(dim12);

#pragma omp for schedule(static)
            for (size_t i = 0; i < n_x; ++i) {
                const quantized_t* xi = x + i * code_size_;

                // Build float LUT from SDC table for this data point's codes
                for (size_t m = 0; m < M_; ++m) {
                    uint8_t qi = GetCode(xi, m);
                    const float* sdc_row = sdc_table_.data() + m * Ks * Ks + qi * Ks;
                    std::memcpy(float_lut.data() + m * Ks, sdc_row, Ks * sizeof(float));
                }

                // Quantize float LUT to uint8
                faiss::quantize_lut::round_uint8_per_column(
                    float_lut.data(), M_, Ks, &cached_norm_a_[i], &cached_norm_b_[i]
                );

                // Cast and store
                uint8_t* out = cached_quant_luts_.data() + i * dim12;
                for (size_t j = 0; j < dim12; ++j) {
                    out[j] = static_cast<uint8_t>(float_lut[j]);
                }
            }
        }

        cached_x_ptr_ = x;
        cached_n_x_ = n_x;
    }

    size_t M_;
    size_t dsub_ = 0;
    size_t d_ = 0;
    size_t code_size_ = 0; // M/2 bytes per vector
    size_t nsq_ = 0;       // = M (already even)
    bool fitted = false;
    std::unique_ptr<faiss::ProductQuantizer> faiss_pq_;
    std::vector<float> sdc_table_; // [M_ × Ks × Ks]

    // LUT cache: quantized LUTs + normalization factors per data point
    mutable const quantized_t* cached_x_ptr_ = nullptr;
    mutable size_t cached_n_x_ = 0;
    mutable std::vector<uint8_t> cached_quant_luts_;  // [n_x × M × Ks]
    mutable std::vector<float> cached_norm_a_;         // [n_x]
    mutable std::vector<float> cached_norm_b_;         // [n_x]
};

} // namespace skmeans
