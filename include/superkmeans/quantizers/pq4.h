#pragma once

#include "superkmeans/common.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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
        SKM_PROFILE_SCOPE("PQ4::fitting");
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
        SKM_PROFILE_SCOPE("PQ4::encoding");
        assert(fitted);
        assert(d == d_);
        faiss_pq_->compute_codes(in, out, n);
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("PQ4::decoding");
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
        SKM_PROFILE_SCOPE("PQ4::search");
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

    void CacheDataPartialNorms(
        const quantized_t* data, size_t n, size_t /*d*/, uint32_t partial_d
    ) override {
        assert(fitted);
        (void)partial_d;
        M_front_ = std::max<size_t>(2, (M_ / 4) & ~1u);
        PrecomputeProgressiveRatios();
        EnsureFrontLUTCache(data, n);
    }

    void CacheCentroidPartialNorms(
        const quantized_t* /*centroids*/, size_t /*n*/, size_t /*d*/, uint32_t /*partial_d*/
    ) override {
        // No-op: centroid packing is done inside FindNearestNeighborWithPruning.
    }

    void FindNearestNeighborWithPruning(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t d,
        uint32_t* out_knn,
        float* out_distances,
        PDXLayout<Quantization::u8, DistanceFunction::l2>& /*pdx_centroids*/,
        uint32_t /*partial_d*/,
        size_t* out_not_pruned_counts
    ) const override {
        SKM_PROFILE_SCOPE("PQ4::FindNearestNeighborWithPruning");
        assert(fitted);
        assert(M_front_ >= 2 && M_front_ <= M_ && M_front_ % 2 == 0);

        EnsureFrontLUTCache(x, n_x);
        EnsureLUTCache(x, n_x); // full LUT needed for batch FastScan on survivors

        // Pack centroids for front subspaces into FastScan block format
        const size_t ntotal2 = ((n_y + BBS - 1) / BBS) * BBS;
        std::vector<uint8_t> packed_front(ntotal2 * M_front_ / 2, 0);
        faiss::pq4_pack_codes(y, n_y, M_, ntotal2, BBS, M_front_, packed_front.data());

        const size_t front_dim12 = Ks * M_front_;
        const size_t full_dim12 = Ks * nsq_;
        const float ad_ratio_front = ComputeADSamplingRatio(M_front_ * dsub_, d);
        const size_t max_surv_padded = ((n_y + BBS - 1) / BBS) * BBS;

#pragma omp parallel num_threads(g_n_threads)
        {
            std::vector<uint8_t> batch_front_lut(NQ_BATCH * front_dim12);
            std::vector<uint8_t> packed_lut(NQ_BATCH * front_dim12);
            std::vector<uint16_t> accu(NQ_BATCH * ntotal2);

            // Per-thread survivor buffers (sized for worst case)
            std::vector<uint32_t> survivor_indices;
            survivor_indices.reserve(n_y);
            std::vector<uint8_t> survivor_codes(n_y * code_size_);
            std::vector<uint8_t> packed_survivors(max_surv_padded * nsq_ / 2);
            std::vector<uint8_t> surv_lut(full_dim12);
            std::vector<uint8_t> packed_surv_lut(full_dim12);
            std::vector<uint16_t> surv_accu(max_surv_padded);

#pragma omp for schedule(dynamic, 8)
            for (size_t i_base = 0; i_base < n_x; i_base += NQ_BATCH) {
                const size_t actual_nq = std::min(NQ_BATCH, n_x - i_base);

                // Phase 1: threshold from previous assignment via exact SDC
                float threshold[NQ_BATCH];
                uint32_t best_j[NQ_BATCH];
                for (size_t q = 0; q < actual_nq; ++q) {
                    const size_t i = i_base + q;
                    best_j[q] = out_knn[i];
                    threshold[q] = ComputeFullSDCDistance(
                        x + i * code_size_, y + best_j[q] * code_size_);
                    out_not_pruned_counts[i] = 0;
                }

                // Phase 2: front FastScan on M_front subspaces
                for (size_t q = 0; q < actual_nq; ++q) {
                    std::memcpy(
                        batch_front_lut.data() + q * front_dim12,
                        cached_front_quant_luts_.data() + (i_base + q) * front_dim12,
                        front_dim12
                    );
                }
                faiss::pq4_pack_LUT(
                    static_cast<int>(actual_nq),
                    static_cast<int>(M_front_),
                    batch_front_lut.data(),
                    packed_lut.data()
                );
                faiss::accumulate_to_mem(
                    static_cast<int>(actual_nq), ntotal2, static_cast<int>(M_front_),
                    packed_front.data(), packed_lut.data(), accu.data()
                );

                // Phase 3: prune via front checkpoint + batch FastScan on survivors
                for (size_t q = 0; q < actual_nq; ++q) {
                    const size_t i = i_base + q;
                    const uint16_t* accu_q = accu.data() + q * ntotal2;
                    const float inv_norm_a = 1.0f / cached_front_norm_a_[i];
                    const float norm_b = cached_front_norm_b_[i];
                    const float thresh_scaled = threshold[q] * ad_ratio_front;

                    // Phase 3a: identify survivors
                    survivor_indices.clear();
                    for (size_t j = 0; j < n_y; ++j) {
                        const float partial = static_cast<float>(accu_q[j]) * inv_norm_a
                                              + norm_b;
                        if (partial <= thresh_scaled) {
                            survivor_indices.push_back(static_cast<uint32_t>(j));
                        }
                    }
                    out_not_pruned_counts[i] = survivor_indices.size();

                    if (survivor_indices.empty()) continue;

                    // Phase 3b: gather survivor codes and pack for FastScan
                    const size_t n_surv = survivor_indices.size();
                    for (size_t s = 0; s < n_surv; ++s) {
                        std::memcpy(
                            survivor_codes.data() + s * code_size_,
                            y + survivor_indices[s] * code_size_,
                            code_size_
                        );
                    }

                    const size_t n_surv_padded = ((n_surv + BBS - 1) / BBS) * BBS;
                    std::memset(packed_survivors.data(), 0, n_surv_padded * nsq_ / 2);
                    faiss::pq4_pack_codes(
                        survivor_codes.data(), n_surv, M_,
                        n_surv_padded, BBS, nsq_, packed_survivors.data()
                    );

                    // Build packed LUT for full M subspaces (from cached full LUT)
                    std::memcpy(
                        surv_lut.data(),
                        cached_quant_luts_.data() + i * full_dim12,
                        full_dim12
                    );
                    faiss::pq4_pack_LUT(
                        1, static_cast<int>(nsq_),
                        surv_lut.data(), packed_surv_lut.data()
                    );

                    // Batch FastScan: all M subspaces on survivors only
                    std::memset(surv_accu.data(), 0, n_surv_padded * sizeof(uint16_t));
                    faiss::accumulate_to_mem(
                        1, n_surv_padded, static_cast<int>(nsq_),
                        packed_survivors.data(), packed_surv_lut.data(),
                        surv_accu.data()
                    );

                    // Phase 3c: find best among survivors
                    const float full_inv_norm_a = 1.0f / cached_norm_a_[i];
                    const float full_norm_b = cached_norm_b_[i];
                    for (size_t s = 0; s < n_surv; ++s) {
                        const float dist = static_cast<float>(surv_accu[s])
                                           * full_inv_norm_a + full_norm_b;
                        if (dist < threshold[q]) {
                            threshold[q] = dist;
                            best_j[q] = survivor_indices[s];
                        }
                    }
                }

                // Write results
                for (size_t q = 0; q < actual_nq; ++q) {
                    out_knn[i_base + q] = best_j[q];
                    out_distances[i_base + q] = threshold[q];
                }
            }
        }
    }

    /**
     * @brief Update centroids via sparse voting with 16×16 SDC tables.
     *
     * Same BLAS sgemm approach as PQ8 but with Ks=16:
     *   votes[n_clusters × 16] = freq[n_clusters × 16] × SDC_m[16 × 16]
     *
     * The voted quantized centroids are decoded to float centroid_accumulators.
     * These are final values (not sums), so FinalizeCentroids is a no-op.
     */
    void UpdateCentroids(
        const quantized_t* encoded_data,
        const uint32_t* assignments,
        float* centroid_accumulators,
        uint32_t* cluster_sizes,
        size_t n, size_t n_clusters, size_t d,
        uint32_t n_threads
    ) const override {
        SKM_PROFILE_SCOPE("PQ4::UpdateCentroids");
        assert(fitted);

        voted_centroids_.resize(n_clusters * code_size_);

        // Count cluster sizes
        std::fill_n(cluster_sizes, n_clusters, 0u);
        for (size_t i = 0; i < n; ++i) {
            cluster_sizes[assignments[i]]++;
        }

        // Phase 1: build all M frequency histograms in one parallel pass.
        // Each thread owns a disjoint cluster range [c0, c1), scans
        // assignments once, and for each owned vector processes all M
        // subspaces back-to-back. The loop inversion amortizes the
        // encoded_data cache miss across M histogram increments, restoring
        // the work/miss ratio that makes the cluster-partition pattern
        // viable for tiny per-i work.
        freq_all_.resize(M_ * n_clusters * Ks);

#pragma omp parallel if (n_threads > 1) num_threads(n_threads)
        {
            uint32_t nt = n_threads;
            uint32_t rank = static_cast<uint32_t>(omp_get_thread_num());
            size_t c0 = (n_clusters * rank) / nt;
            size_t c1 = (n_clusters * (rank + 1)) / nt;

            for (size_t m = 0; m < M_; ++m) {
                std::fill_n(
                    freq_all_.data() + m * n_clusters * Ks + c0 * Ks,
                    (c1 - c0) * Ks, 0.0f
                );
            }

            for (size_t i = 0; i < n; ++i) {
                uint32_t c = assignments[i];
                if (c < c0 || c >= c1) continue;
                const quantized_t* xi = encoded_data + i * code_size_;
                for (size_t mb = 0; mb < code_size_; ++mb) {
                    uint8_t b = xi[mb];
                    const size_t m0 = 2 * mb;
                    const size_t m1 = m0 + 1;
                    freq_all_[m0 * n_clusters * Ks + c * Ks + (b & 0xF)] += 1.0f;
                    freq_all_[m1 * n_clusters * Ks + c * Ks + (b >> 4)] += 1.0f;
                }
            }
        }

        // Phase 2: parallel over byte position mb. Each thread fully owns
        // byte mb of voted_centroids_ across all clusters (low nibble = even
        // subspace 2*mb, high nibble = odd subspace 2*mb+1), runs both
        // sgemms for that byte's subspace pair, and packs the nibble pair
        // directly. No packing race, no fill_n of voted_centroids_ needed.
#pragma omp parallel if (n_threads > 1) num_threads(n_threads)
        {
            std::vector<float> votes_even(n_clusters * Ks);
            std::vector<float> votes_odd(n_clusters * Ks);

#pragma omp for schedule(dynamic)
            for (size_t mb = 0; mb < code_size_; ++mb) {
                float* votes_targets[2] = {votes_even.data(), votes_odd.data()};
                for (int parity = 0; parity < 2; ++parity) {
                    const size_t m = 2 * mb + parity;
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
                        freq_all_.data() + m * n_clusters * Ks, &ldb,
                        &beta_val,
                        votes_targets[parity], &ldc
                    );
                }

                for (size_t c = 0; c < n_clusters; ++c) {
                    uint8_t lo = 0, hi = 0;
                    if (cluster_sizes[c] != 0) {
                        const float* row_lo = votes_even.data() + c * Ks;
                        const float* row_hi = votes_odd.data() + c * Ks;
                        size_t best_lo = 0, best_hi = 0;
                        float v_lo = row_lo[0], v_hi = row_hi[0];
                        for (size_t k = 1; k < Ks; ++k) {
                            if (row_lo[k] < v_lo) { v_lo = row_lo[k]; best_lo = k; }
                            if (row_hi[k] < v_hi) { v_hi = row_hi[k]; best_hi = k; }
                        }
                        lo = static_cast<uint8_t>(best_lo);
                        hi = static_cast<uint8_t>(best_hi);
                    }
                    voted_centroids_[c * code_size_ + mb] =
                        static_cast<uint8_t>((hi << 4) | (lo & 0x0F));
                }
            }
        }

        // Decode voted quantized centroids to float
        faiss_pq_->decode(voted_centroids_.data(), centroid_accumulators, n_clusters);
    }

    void FinalizeCentroids(
        float* /*centroids*/,
        const uint32_t* /*cluster_sizes*/,
        size_t /*n_clusters*/, size_t /*d*/
    ) const override {
        // No-op: UpdateCentroids already produced final float centroids
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return true; }
    bool NeedsPDXLayout() const override { return false; }

    /// ~12.5% of d, aligned to 8 for byte-aligned PQ code boundaries.
    uint32_t InitialPartialD(uint32_t vertical_d) const override {
        return std::max<uint32_t>(MIN_PARTIAL_D, ((vertical_d / 8) + 7) & ~7u);
    }

    /// Round up to multiple of 8 after tuning adjustment.
    uint32_t AlignPartialD(uint32_t partial_d, uint32_t vertical_d) const override {
        return std::min((partial_d + 7) & ~7u, vertical_d);
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

    void EnsureFrontLUTCache(const quantized_t* x, size_t n_x) const {
        if (cached_front_x_ptr_ == x && cached_front_n_x_ == n_x
            && cached_front_M_ == M_front_) return;

        const size_t front_dim12 = Ks * M_front_;
        cached_front_quant_luts_.resize(n_x * front_dim12);
        cached_front_norm_a_.resize(n_x);
        cached_front_norm_b_.resize(n_x);

#pragma omp parallel num_threads(g_n_threads)
        {
            std::vector<float> float_lut(front_dim12);

#pragma omp for schedule(static)
            for (size_t i = 0; i < n_x; ++i) {
                const quantized_t* xi = x + i * code_size_;

                for (size_t m = 0; m < M_front_; ++m) {
                    uint8_t qi = GetCode(xi, m);
                    const float* sdc_row = sdc_table_.data() + m * Ks * Ks + qi * Ks;
                    std::memcpy(float_lut.data() + m * Ks, sdc_row, Ks * sizeof(float));
                }

                faiss::quantize_lut::round_uint8_per_column(
                    float_lut.data(), M_front_, Ks,
                    &cached_front_norm_a_[i], &cached_front_norm_b_[i]
                );

                uint8_t* out = cached_front_quant_luts_.data() + i * front_dim12;
                for (size_t j = 0; j < front_dim12; ++j) {
                    out[j] = static_cast<uint8_t>(float_lut[j]);
                }
            }
        }

        cached_front_x_ptr_ = x;
        cached_front_n_x_ = n_x;
        cached_front_M_ = M_front_;
    }

    float ComputeFullSDCDistance(
        const quantized_t* xi, const quantized_t* yj
    ) const {
        // SKM_PROFILE_SCOPE("PQ4::full_sdc_distance");
        float dist = 0.0f;
        for (size_t m = 0; m < M_; ++m) {
            dist += sdc_table_[m * Ks * Ks + GetCode(xi, m) * Ks + GetCode(yj, m)];
        }
        return dist;
    }

    static constexpr size_t PROGRESSIVE_STEP = 16;

    /// Precompute ADSampling ratios for progressive checkpoints starting after M_front_.
    void PrecomputeProgressiveRatios() {
        progressive_ratios_.clear();
        for (size_t m = M_front_ + PROGRESSIVE_STEP; m < M_; m += PROGRESSIVE_STEP) {
            progressive_ratios_.push_back(ComputeADSamplingRatio(m * dsub_, d_));
        }
        std::cout << progressive_ratios_[0] << "\n";
    }

    float ComputeProgressiveSDCDistance(
        const quantized_t* xi, const quantized_t* yj,
        float threshold, float front_dist
    ) const {
        // SKM_PROFILE_SCOPE("PQ4::progressive_sdc_distance"); 
        float dist = front_dist;
        size_t m = M_front_;
        size_t ratio_idx = 0;

        // Full chunks of PROGRESSIVE_STEP subspaces with 1 branch per chunk
        for (; m + PROGRESSIVE_STEP <= M_; m += PROGRESSIVE_STEP) {
            for (size_t k = 0; k < PROGRESSIVE_STEP; ++k) {
                const size_t sub = m + k;
                dist += sdc_table_[sub * Ks * Ks + GetCode(xi, sub) * Ks + GetCode(yj, sub)];
            }
            if (m + PROGRESSIVE_STEP < M_) {
                if (dist > threshold * progressive_ratios_[ratio_idx++]){
                    return std::numeric_limits<float>::max();
                }
            }
        }
        // Remainder (< PROGRESSIVE_STEP subspaces)
        for (; m < M_; ++m) {
            dist += sdc_table_[m * Ks * Ks + GetCode(xi, m) * Ks + GetCode(yj, m)];
        }
        return dist;
    }

    static float ComputeADSamplingRatio(size_t front_d, size_t d) {
        if (front_d == 0 || front_d >= d) return 1.0f;
        const double eps0 = static_cast<double>(PRUNER_INITIAL_THRESHOLD);
        const double ratio =
            static_cast<double>(front_d) / static_cast<double>(d) *
            (1.0 + eps0 / std::sqrt(static_cast<double>(front_d))) *
            (1.0 + eps0 / std::sqrt(static_cast<double>(front_d)));
        return static_cast<float>(ratio);
    }

    size_t M_;
    size_t dsub_ = 0;
    size_t d_ = 0;
    size_t code_size_ = 0; // M/2 bytes per vector
    size_t nsq_ = 0;       // = M (already even)
    bool fitted = false;
    std::unique_ptr<faiss::ProductQuantizer> faiss_pq_;
    std::vector<float> sdc_table_; // [M_ × Ks × Ks]
    std::vector<float> progressive_ratios_; // precomputed ADSampling ratios per chunk
    mutable std::vector<quantized_t> voted_centroids_; // [n_clusters × code_size_]
    mutable std::vector<float> freq_all_;              // [M_ × n_clusters × Ks]

    // LUT cache: quantized LUTs + normalization factors per data point
    mutable const quantized_t* cached_x_ptr_ = nullptr;
    mutable size_t cached_n_x_ = 0;
    mutable std::vector<uint8_t> cached_quant_luts_;  // [n_x × M × Ks]
    mutable std::vector<float> cached_norm_a_;         // [n_x]
    mutable std::vector<float> cached_norm_b_;         // [n_x]

    // Front LUT cache for pruning
    mutable size_t M_front_ = 0;
    mutable size_t cached_front_M_ = 0;
    mutable const quantized_t* cached_front_x_ptr_ = nullptr;
    mutable size_t cached_front_n_x_ = 0;
    mutable std::vector<uint8_t> cached_front_quant_luts_;  // [n_x × M_front × Ks]
    mutable std::vector<float> cached_front_norm_a_;         // [n_x]
    mutable std::vector<float> cached_front_norm_b_;         // [n_x]
};

} // namespace skmeans
