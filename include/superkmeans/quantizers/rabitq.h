#pragma once

// #ifdef HAS_FAISS

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <vector>

#include <faiss/impl/RaBitQuantizer.h>

namespace skmeans {

struct RaBitQFactors {
    float or_minus_c_l2sqr; // ||original - centroid||²  (for L2 metric)
    float dp_multiplier;    // scaling factor for dot-product estimation
};
static_assert(sizeof(RaBitQFactors) == 8, "RaBitQFactors must match FAISS FactorsData");

/**
 * @brief RaBitQ quantizer with FastScan-accelerated distance kernel.
 *
 * Uses FAISS RaBitQuantizer for Fit/Encode/Decode.
 * Distance computation uses a custom FastScan kernel (VPSHUFB/TBL lookups)
 * that processes 32 data points simultaneously per centroid.
 *
 * For k-means: centroids (K, small) are SQ-quantized to qb bits once,
 * then LUTs are built from the quantized values. Data points (N, large)
 * are byte-transposed into blocks of 32 for SIMD lookup.
 */
class RaBitQQuantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;

    void Fit(const float* data, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("RaBitQ::Fit");
        assert(d % 8 == 0 && "RaBitQ-GEMM requires dimensionality divisible by 8");
        d_ = d;
        binary_bytes_ = (d + 7) / 8;
        centroid_.resize(d, 0.0f);

        // Compute dataset mean as the centering centroid for RaBitQ
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < d; ++j) {
            double sum = 0;
            for (size_t i = 0; i < n; ++i) {
                sum += data[i * d + j];
            }
            centroid_[j] = static_cast<float>(sum / static_cast<double>(n));
        }

        faiss_quantizer_ = std::make_unique<faiss::RaBitQuantizer>(d, faiss::METRIC_L2);
        faiss_quantizer_->centroid = centroid_.data();
        faiss_code_size_ = faiss_quantizer_->code_size;
        assert(faiss_code_size_ == CodeSize(d));
        fitted_ = true;
    }

    void Encode(const float* in, quantized_t* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("RaBitQ::Encode");
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->compute_codes(in, reinterpret_cast<uint8_t*>(out), n);
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("RaBitQ::Decode");
        assert(fitted_);
        assert(d == d_);
        faiss_quantizer_->decode(reinterpret_cast<const uint8_t*>(in), out, n);
    }

    void ComputeNorms(
        const quantized_t* data, size_t n, size_t d, float* out_norms
    ) const override {
        SKM_PROFILE_SCOPE("RaBitQ::ComputeNorms");

        assert(fitted_);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code =
                reinterpret_cast<const uint8_t*>(data) + i * faiss_code_size_;
            const auto* factors =
                reinterpret_cast<const RaBitQFactors*>(code + binary_bytes_);
            out_norms[i] = factors->or_minus_c_l2sqr;
        }
    }

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
        SKM_PROFILE_SCOPE("RaBitQ::Search::FastScanDistance");
        assert(fitted_);
        (void)y;
        (void)x_float;
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);

        // Cache per-data-point factors (depend only on x, reused across iterations)
        EnsureCodeFactorsCache(x_codes, n_x);
        EnsureTransposedBlocksCache(x_codes, n_x);
        const uint32_t* sum_q = cached_sum_q_.data();
        const float* or_c_l2sqr = cached_or_c_l2sqr_.data();
        const float* dp_mult = cached_dp_mult_.data();

        // Quantize centroids and build LUTs (changes every iteration)
        const size_t n_sub = 2 * binary_bytes_; // 2 sub-quantizers per byte position
        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        std::vector<uint8_t> all_luts(n_y * n_sub * 16);
        QuantizeCentroidsAndBuildLUTs(
            y_float, n_y, d,
            all_luts.data(), c1.data(), c2.data(), c34.data(), qr_to_c_l2sqr.data()
        );

        const size_t lut_stride = n_sub * 16;
        const size_t block_bytes = FastScanComputer::kBlockSize * binary_bytes_;
        const size_t n_blocks = cached_n_blocks_;

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        {
            SKM_PROFILE_SCOPE("RaBitQ::Search::FastScanDistance");
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t blk = 0; blk < n_blocks; ++blk) {
                const size_t blk_start = blk * FastScanComputer::kBlockSize;
                const size_t blk_count = std::min(FastScanComputer::kBlockSize, n_x - blk_start);

                const uint8_t* packed = cached_transposed_.get() + blk * block_bytes;

                float best_dist[FastScanComputer::kBlockSize];
                uint32_t best_idx[FastScanComputer::kBlockSize];
                std::fill_n(best_dist, FastScanComputer::kBlockSize, std::numeric_limits<float>::max());
                std::fill_n(best_idx, FastScanComputer::kBlockSize, 0u);

                float dist_buf[FastScanComputer::kBlockSize];

                // Precompute sum_q as float once per block (avoids u32→f32 per centroid)
                float sum_q_f32[FastScanComputer::kBlockSize];
                for (size_t k = 0; k < blk_count; ++k) {
                    sum_q_f32[k] = static_cast<float>(sum_q[blk_start + k]);
                }

                for (size_t j = 0; j < n_y; ++j) {
                    uint16_t dot_qo[FastScanComputer::kBlockSize];
                    FastScanComputer::ScanBlock(
                        packed, all_luts.data() + j * lut_stride,
                        binary_bytes_, dot_qo, blk_count
                    );

                    FastScanComputer::RabitQCorrection(
                        dot_qo,
                        c1[j], c2[j], c34[j], qr_to_c_l2sqr[j],
                        sum_q_f32,
                        or_c_l2sqr + blk_start,
                        dp_mult + blk_start,
                        dist_buf,
                        blk_count
                    );

                    for (size_t k = 0; k < blk_count; ++k) {
                        if (dist_buf[k] < best_dist[k]) {
                            best_dist[k] = dist_buf[k];
                            best_idx[k] = static_cast<uint32_t>(j);
                        }
                    }
                }

                for (size_t k = 0; k < blk_count; ++k) {
                    out_distances[blk_start + k] = best_dist[k];
                    out_knn[blk_start + k] = best_idx[k];
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
        assert(fitted_);
        (void)y_quantized;
        (void)norms_x;
        (void)norms_y;
        (void)tmp_buf;
    }

    size_t CodeSize(size_t d) const override {
        return (d + 7) / 8 + sizeof(RaBitQFactors);
    }

    bool IsFitted() const override { return fitted_; }
    bool SupportsPruning() const override { return true; }
    bool NeedsPDXLayout() const override { return false; }

    void CacheDataPartialNorms(
        const quantized_t* data, size_t n, size_t /*d*/, uint32_t partial_d
    ) override {
        SKM_PROFILE_SCOPE("RaBitQ::CacheDataPartialNorms");
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(data);
        const size_t front_bytes = partial_d / 8;
        const size_t mid_bytes = d_ / 32;  // d/4, byte-aligned
        cached_sum_q_front_.resize(n);
        cached_sum_q_mid_.resize(n);
        cached_partial_d_ = partial_d;
        pruning_partial_norms_dirty_ = true;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes + i * faiss_code_size_;
            uint32_t pc = 0;
            size_t b = 0;
            for (; b < std::min(front_bytes, mid_bytes); ++b) {
                pc += static_cast<uint32_t>(__builtin_popcount(code[b]));
            }
            if (front_bytes <= mid_bytes) {
                cached_sum_q_front_[i] = pc;
                for (; b < mid_bytes; ++b) {
                    pc += static_cast<uint32_t>(__builtin_popcount(code[b]));
                }
                cached_sum_q_mid_[i] = pc;
            } else {
                cached_sum_q_mid_[i] = pc;
                for (; b < front_bytes; ++b) {
                    pc += static_cast<uint32_t>(__builtin_popcount(code[b]));
                }
                cached_sum_q_front_[i] = pc;
            }
        }
    }

    void CacheCentroidPartialNorms(
        const quantized_t* /*centroids*/, size_t /*n*/, size_t /*d*/, uint32_t /*partial_d*/
    ) override {
        // No-op: centroid-side partial values are computed inside
        // FindNearestNeighborWithPruning during LUT building.
    }

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
        PDXLayout<Quantization::u8, DistanceFunction::l2>& /*pdx_centroids*/,
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) const override {
        SKM_PROFILE_SCOPE("RaBitQ::FindNearestNeighborWithPruning");
        assert(fitted_);
        (void)y;

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);

        // Ensure data-side caches
        EnsureCodeFactorsCache(x_codes, n_x);
        EnsureTransposedBlocksCache(x_codes, n_x);
        EnsurePartialNormsCache(x_float, n_x, d, partial_d);

        const uint32_t* sum_q = cached_sum_q_.data();
        const float* or_c_l2sqr = cached_or_c_l2sqr_.data();
        const float* dp_mult = cached_dp_mult_.data();
        const uint32_t* sum_q_front = cached_sum_q_front_.data();
        const float* or_c_l2sqr_front = cached_or_c_l2sqr_front_.data();
        const uint32_t* sum_q_mid = cached_sum_q_mid_.data();
        const float* or_c_l2sqr_mid = cached_or_c_l2sqr_mid_.data();

        // Quantize centroids and build LUTs with partial bounds
        const size_t front_bytes = partial_d / 8;
        const size_t front_d = front_bytes * 8;
        const size_t mid_bytes = d / 32;  // d/4, byte-aligned
        const size_t mid_d = mid_bytes * 8;
        const bool use_mid_checkpoint = (front_bytes < mid_bytes) && (mid_bytes < binary_bytes_);
        const size_t gap_bytes = use_mid_checkpoint ? (mid_bytes - front_bytes) : 0;
        const size_t phase3_start = use_mid_checkpoint ? mid_bytes : front_bytes;
        const size_t phase3_bytes = binary_bytes_ - phase3_start;
        const size_t n_sub = 2 * binary_bytes_;

        std::vector<float> c1(n_y), c2(n_y), c34(n_y), qr_to_c_l2sqr(n_y);
        std::vector<float> qr_to_c_l2sqr_front(n_y), c34_front(n_y);
        std::vector<float> qr_to_c_l2sqr_mid(n_y), c34_mid(n_y);
        std::vector<uint8_t> all_luts(n_y * n_sub * 16);
        std::vector<uint8_t> centroid_planes(qb_ * n_y * binary_bytes_, 0);
        QuantizeCentroidsAndBuildLUTsWithBounds(
            y_float, n_y, d, front_d, mid_d,
            all_luts.data(), c1.data(), c2.data(), c34.data(), qr_to_c_l2sqr.data(),
            qr_to_c_l2sqr_front.data(), c34_front.data(),
            qr_to_c_l2sqr_mid.data(), c34_mid.data(),
            centroid_planes.data()
        );

        // ADSampling ratios
        const float adsampling_ratio_front = ComputeADSamplingRatio(front_d, d);
        const float adsampling_ratio_mid = use_mid_checkpoint
            ? ComputeADSamplingRatio(mid_d, d) : 1.0f;

        const size_t lut_stride = n_sub * 16;
        const size_t block_bytes = FastScanComputer::kBlockSize * binary_bytes_;
        const size_t n_blocks = cached_n_blocks_;

        using b8_computer = DistanceComputer<DistanceFunction::l2, Quantization::b8>;

        {
            SKM_PROFILE_SCOPE("RaBitQ::PrunedScan");
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t blk = 0; blk < n_blocks; ++blk) {
                const size_t blk_start = blk * FastScanComputer::kBlockSize;
                const size_t blk_count =
                    std::min(FastScanComputer::kBlockSize, n_x - blk_start);

                const uint8_t* packed = cached_transposed_.get() + blk * block_bytes;

                float best_dist[FastScanComputer::kBlockSize];
                uint32_t best_idx[FastScanComputer::kBlockSize];

                // Phase 1: threshold from previous centroid via per-pair LUT lookup
                for (size_t k = 0; k < blk_count; ++k) {
                    const size_t i = blk_start + k;
                    const uint32_t prev_j = out_knn[i];
                    best_idx[k] = prev_j;
                    best_dist[k] = ComputeFullDistanceViaLUT(
                        x_codes + i * faiss_code_size_,
                        all_luts.data() + prev_j * lut_stride,
                        c1[prev_j], c2[prev_j], c34[prev_j],
                        qr_to_c_l2sqr[prev_j],
                        sum_q[i], or_c_l2sqr[i], dp_mult[i]
                    );
                    out_not_pruned_counts[i] = 0;
                }

                // ── Pass 1a: FastScan all centroids ──
                std::vector<uint16_t> all_partial_dots(n_y * FastScanComputer::kBlockSize);

                {
                    SKM_PROFILE_SCOPE("RaBitQ::PrunedScan/fastscan");
                    for (size_t j = 0; j < n_y; ++j) {
                        FastScanComputer::ScanBlock(
                            packed, all_luts.data() + j * lut_stride,
                            front_bytes,
                            all_partial_dots.data() + j * FastScanComputer::kBlockSize,
                            blk_count
                        );
                    }
                }

                // ── Pass 1b: SIMD correction + checkpoint 1 compaction ──
                float partial_l2_buf[FastScanComputer::kBlockSize];

                // Precompute per-block buffers (avoids redundant ops per centroid j)
                float sum_q_front_f32[FastScanComputer::kBlockSize];
                float threshold_buf[FastScanComputer::kBlockSize];
                for (size_t k = 0; k < blk_count; ++k) {
                    sum_q_front_f32[k] = static_cast<float>(sum_q_front[blk_start + k]);
                    threshold_buf[k] = best_dist[k] * adsampling_ratio_front;
                }

                // Flat survivor buffers: separate k and j arrays
                const size_t max_survivors = blk_count * n_y;
                std::unique_ptr<uint32_t[]> survivor_ks(new uint32_t[max_survivors]);
                std::unique_ptr<uint32_t[]> survivor_js(new uint32_t[max_survivors]);
                size_t total_survivors = 0;

                {
                    SKM_PROFILE_SCOPE("RaBitQ::PrunedScan/checkpoint1");
                    for (size_t j = 0; j < n_y; ++j) {
                        const uint16_t* partial_dot_qo =
                            all_partial_dots.data() + j * FastScanComputer::kBlockSize;

                        // SIMD float correction for all 32 points
                        FastScanComputer::RabitQCorrection(
                            partial_dot_qo,
                            c1[j], c2[j], c34_front[j], qr_to_c_l2sqr_front[j],
                            sum_q_front_f32,
                            or_c_l2sqr_front + blk_start,
                            dp_mult + blk_start,
                            partial_l2_buf,
                            blk_count
                        );

                        // SIMD compaction directly into survivor_ks
                        size_t n_new = 0;
                        FastScanComputer::RabitQCompactSurvivors(
                            blk_count, n_new,
                            survivor_ks.get() + total_survivors,
                            partial_l2_buf, threshold_buf
                        );

                        std::fill_n(survivor_js.get() + total_survivors, n_new, static_cast<uint32_t>(j));

                        for (size_t s = 0; s < n_new; ++s) {
                            out_not_pruned_counts[blk_start + survivor_ks[total_survivors + s]]++;
                        }
                        total_survivors += n_new;
                    }
                }

                // ── Pass 2: checkpoint 2 + Phase 3 (centroid-grouped) ──
                // Survivors are sorted by j. Process groups per centroid.
                {
                    SKM_PROFILE_SCOPE("RaBitQ::PrunedScan/remaining");
                    uint32_t accumulated_dots[FastScanComputer::kBlockSize];
                    float correction_buf[FastScanComputer::kBlockSize];

                    // Precompute sum_q as float once per block
                    float sum_q_mid_f32[FastScanComputer::kBlockSize];
                    float sum_q_f32[FastScanComputer::kBlockSize];
                    for (size_t k = 0; k < blk_count; ++k) {
                        sum_q_mid_f32[k] = static_cast<float>(sum_q_mid[blk_start + k]);
                        sum_q_f32[k] = static_cast<float>(sum_q[blk_start + k]);
                    }

                    size_t s = 0;
                    while (s < total_survivors) {
                        const uint32_t j = survivor_js[s];
                        const size_t group_start = s;
                        while (s < total_survivors && survivor_js[s] == j) ++s;

                        // Initialize accumulated dots from front partial dots
                        const uint16_t* partial_dot_row =
                            all_partial_dots.data() + j * FastScanComputer::kBlockSize;
                        for (size_t k = 0; k < blk_count; ++k) {
                            accumulated_dots[k] = static_cast<uint32_t>(partial_dot_row[k]);
                        }

                        size_t phase3_start_idx = group_start;
                        size_t phase3_end_idx = s;

                        // Checkpoint 2: extend to mid_d dims
                        if (use_mid_checkpoint) {
                            for (size_t si = group_start; si < s; ++si) {
                                const size_t k = survivor_ks[si];
                                const size_t i = blk_start + k;
                                const uint8_t* data_code = x_codes + i * faiss_code_size_;
                                uint32_t gap_dot = 0;
                                for (int bp = 0; bp < qb_; ++bp) {
                                    const uint8_t* plane = centroid_planes.data() +
                                        (bp * n_y + j) * binary_bytes_ + front_bytes;
                                    gap_dot +=
                                        b8_computer::Horizontal(
                                            data_code + front_bytes, plane, gap_bytes
                                        ) << bp;
                                }
                                accumulated_dots[k] += gap_dot;
                            }

                            // SIMD correction for mid checkpoint (full block)
                            FastScanComputer::RabitQCorrectionU32(
                                accumulated_dots,
                                c1[j], c2[j], c34_mid[j], qr_to_c_l2sqr_mid[j],
                                sum_q_mid_f32,
                                or_c_l2sqr_mid + blk_start,
                                dp_mult + blk_start,
                                correction_buf,
                                blk_count
                            );

                            // Compact survivors that pass mid threshold
                            size_t write = group_start;
                            for (size_t si = group_start; si < s; ++si) {
                                const uint32_t k = survivor_ks[si];
                                if (correction_buf[k] <= best_dist[k] * adsampling_ratio_mid) {
                                    survivor_ks[write] = survivor_ks[si];
                                    write++;
                                }
                            }
                            phase3_start_idx = group_start;
                            phase3_end_idx = write;
                        }

                        // Phase 3: add remaining popcount for survivors
                        for (size_t si = phase3_start_idx; si < phase3_end_idx; ++si) {
                            const size_t k = survivor_ks[si];
                            const size_t i = blk_start + k;
                            const uint8_t* data_code = x_codes + i * faiss_code_size_;
                            uint32_t remaining_dot_qo = 0;
                            for (int bp = 0; bp < qb_; ++bp) {
                                const uint8_t* plane = centroid_planes.data() +
                                    (bp * n_y + j) * binary_bytes_ + phase3_start;
                                remaining_dot_qo +=
                                    b8_computer::Horizontal(
                                        data_code + phase3_start, plane, phase3_bytes
                                    ) << bp;
                            }
                            accumulated_dots[k] += remaining_dot_qo;
                        }

                        // SIMD correction for final distance (full block)
                        FastScanComputer::RabitQCorrectionU32(
                            accumulated_dots,
                            c1[j], c2[j], c34[j], qr_to_c_l2sqr[j],
                            sum_q_f32,
                            or_c_l2sqr + blk_start,
                            dp_mult + blk_start,
                            correction_buf,
                            blk_count
                        );

                        // Update best for phase 3 survivors
                        for (size_t si = phase3_start_idx; si < phase3_end_idx; ++si) {
                            const uint32_t k = survivor_ks[si];
                            if (correction_buf[k] < best_dist[k]) {
                                best_dist[k] = correction_buf[k];
                                best_idx[k] = j;
                            }
                        }
                    }
                }

                for (size_t k = 0; k < blk_count; ++k) {
                    out_distances[blk_start + k] = best_dist[k];
                    out_knn[blk_start + k] = best_idx[k];
                }
            }
        }
    }

  private:
    /// Extract per-code metadata: popcount, or_c_l2sqr, dp_multiplier.
    void PrecomputeCodeFactors(
        const uint8_t* codes, size_t n,
        uint32_t* sum_q, float* or_c_l2sqr, float* dp_mult
    ) const {
        SKM_PROFILE_SCOPE("RaBitQ::PrecomputeCodeFactors");
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes + i * faiss_code_size_;

            // Popcount of the binary part
            uint32_t pc = 0;
            size_t b = 0;
            for (; b + 8 <= binary_bytes_; b += 8) {
                uint64_t word;
                std::memcpy(&word, code + b, 8);
                pc += static_cast<uint32_t>(__builtin_popcountll(word));
            }
            for (; b < binary_bytes_; ++b) {
                pc += static_cast<uint32_t>(__builtin_popcount(code[b]));
            }
            sum_q[i] = pc;

            const auto* fac = reinterpret_cast<const RaBitQFactors*>(code + binary_bytes_);
            or_c_l2sqr[i] = fac->or_minus_c_l2sqr;
            dp_mult[i] = fac->dp_multiplier;
        }
    }

    /// SQ-quantize centroids to qb bits and build 16-entry LUTs for FastScan.
    ///
    /// Each centroid produces d SQ-quantized uint8 values.
    /// Sub-quantizer m corresponds to 4 consecutive dimensions (4m..4m+3).
    /// LUT[m][c] = sum of quantized values at dimensions where bit k is set in c.
    ///
    /// Layout: lut[j * n_sub * 16 + m * 16 + c] for centroid j, sub-quantizer m, code c.
    /// Sub-quantizers are ordered: byte 0 low nibble, byte 0 high nibble,
    ///                             byte 1 low nibble, byte 1 high nibble, ...
    void QuantizeCentroidsAndBuildLUTs(
        const float* y_float, size_t n_y, size_t d,
        uint8_t* all_luts,
        float* c1, float* c2, float* c34, float* qr_to_c_l2sqr
    ) const {
        SKM_PROFILE_SCOPE("RaBitQ::QuantizeCentroidsAndBuildLUTs");
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const float max_val = static_cast<float>((1 << qb_) - 1);
        const size_t n_sub = 2 * binary_bytes_;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < n_y; ++j) {
            std::vector<float> rotated(d);
            float v_min = std::numeric_limits<float>::max();
            float v_max = std::numeric_limits<float>::lowest();
            float norm_sq = 0;

            for (size_t dim = 0; dim < d; ++dim) {
                rotated[dim] = y_float[j * d + dim] - centroid_[dim];
                v_min = std::min(v_min, rotated[dim]);
                v_max = std::max(v_max, rotated[dim]);
                norm_sq += rotated[dim] * rotated[dim];
            }
            qr_to_c_l2sqr[j] = norm_sq;

            float delta = (v_max - v_min) / max_val;
            if (delta < std::numeric_limits<float>::epsilon()) delta = 1.0f;
            const float inv_delta = 1.0f / delta;
            float sum_qq = 0;

            std::vector<uint8_t> quantized(d);
            for (size_t dim = 0; dim < d; ++dim) {
                int v = static_cast<int>(std::lround((rotated[dim] - v_min) * inv_delta));
                v = std::max(0, std::min(v, static_cast<int>(max_val)));
                quantized[dim] = static_cast<uint8_t>(v);
                sum_qq += static_cast<float>(v);
            }

            c1[j] = 2.0f * delta * inv_sqrt_d;
            c2[j] = 2.0f * v_min * inv_sqrt_d;
            c34[j] = inv_sqrt_d * (delta * sum_qq + static_cast<float>(d) * v_min);

            // Build LUTs for this centroid
            // Each byte of the binary code maps to 2 sub-quantizers (4 dims each).
            // Sub-quantizer 2b handles dims 8b..8b+3 (low nibble of byte b)
            // Sub-quantizer 2b+1 handles dims 8b+4..8b+7 (high nibble of byte b)
            uint8_t* lut_j = all_luts + j * n_sub * 16;
            for (size_t b = 0; b < binary_bytes_; ++b) {
                uint8_t* lut_lo = lut_j + (2 * b) * 16;
                uint8_t* lut_hi = lut_j + (2 * b + 1) * 16;

                // Get the 8 SQ values for dimensions 8b..8b+7
                uint8_t sq[8] = {0};
                for (int k = 0; k < 8 && (8 * b + k) < d; ++k) {
                    sq[k] = quantized[8 * b + k];
                }

                // Low nibble LUT: dims 8b..8b+3
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1) val += sq[0];
                    if (c & 2) val += sq[1];
                    if (c & 4) val += sq[2];
                    if (c & 8) val += sq[3];
                    lut_lo[c] = val;
                }

                // High nibble LUT: dims 8b+4..8b+7
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1) val += sq[4];
                    if (c & 2) val += sq[5];
                    if (c & 4) val += sq[6];
                    if (c & 8) val += sq[7];
                    lut_hi[c] = val;
                }
            }
        }
    }

    /// Nibble-split transpose with kPerm0 interleaving for fast SIMD scanning.
    ///
    /// For each byte position b, output layout (32 bytes):
    ///   out[0..15]:  lo nibbles, kPerm0-interleaved, 2 vectors per byte
    ///   out[16..31]: hi nibbles, kPerm0-interleaved, 2 vectors per byte
    ///
    /// This allows a single 256-bit LUT load (lo_LUT|hi_LUT) and
    /// vpshufb per-lane to match codes to correct LUTs without broadcasts.
    static constexpr int kPerm0[16] = {
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
    };

    void TransposeBlock(
        const uint8_t* codes, size_t blk_start, size_t blk_count,
        uint8_t* packed
    ) const {
        std::memset(packed, 0, binary_bytes_ * FastScanComputer::kBlockSize);

        for (size_t b = 0; b < binary_bytes_; ++b) {
            uint8_t* out = packed + b * FastScanComputer::kBlockSize;

            // Gather byte b from each of the 32 vectors
            uint8_t col[32] = {0};
            for (size_t k = 0; k < blk_count; ++k) {
                col[k] = codes[(blk_start + k) * faiss_code_size_ + b];
            }

            for (int j = 0; j < 16; ++j) {
                const int vA = kPerm0[j];
                const int vB = kPerm0[j] + 16;
                out[j]      = (col[vA] & 0x0F) | ((col[vB] & 0x0F) << 4);
                out[j + 16] = (col[vA] >> 4)    | ((col[vB] >> 4) << 4);
            }
        }
    }

    /// Cache PrecomputeCodeFactors results (depend only on x, not centroids).
    void EnsureCodeFactorsCache(const uint8_t* x_codes, size_t n_x) const {
        SKM_PROFILE_SCOPE("RaBitQ::EnsureCodeFactorsCache");
        if (cached_x_ptr_ == x_codes && cached_n_x_ == n_x) return;

        cached_sum_q_.resize(n_x);
        cached_or_c_l2sqr_.resize(n_x);
        cached_dp_mult_.resize(n_x);
        PrecomputeCodeFactors(
            x_codes, n_x,
            cached_sum_q_.data(), cached_or_c_l2sqr_.data(), cached_dp_mult_.data()
        );
        cached_x_ptr_ = x_codes;
        cached_n_x_ = n_x;
    }

    int qb_ = 4;
    size_t d_ = 0;
    size_t binary_bytes_ = 0;
    size_t faiss_code_size_ = 0;
    std::vector<float> centroid_;
    std::unique_ptr<faiss::RaBitQuantizer> faiss_quantizer_;
    bool fitted_ = false;

    // Cached per-data-point factors (reused across iterations)
    mutable const uint8_t* cached_x_ptr_ = nullptr;
    mutable size_t cached_n_x_ = 0;
    mutable std::vector<uint32_t> cached_sum_q_;
    mutable std::vector<float> cached_or_c_l2sqr_;
    mutable std::vector<float> cached_dp_mult_;

    // Cached transposed blocks (reused across iterations)
    mutable std::unique_ptr<uint8_t[]> cached_transposed_;
    mutable size_t cached_n_blocks_ = 0;

    /// Transpose all data blocks once and cache for reuse across iterations.
    void EnsureTransposedBlocksCache(const uint8_t* x_codes, size_t n_x) const {
        SKM_PROFILE_SCOPE("RaBitQ::EnsureTransposedBlocksCache");
        // Piggyback on the same pointer check as code factors
        if (cached_n_blocks_ > 0 && cached_x_ptr_ == x_codes && cached_n_x_ == n_x) return;

        const size_t block_bytes = FastScanComputer::kBlockSize * binary_bytes_;
        cached_n_blocks_ = (n_x + FastScanComputer::kBlockSize - 1) / FastScanComputer::kBlockSize;
        cached_transposed_.reset(new uint8_t[cached_n_blocks_ * block_bytes]);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t blk = 0; blk < cached_n_blocks_; ++blk) {
            const size_t blk_start = blk * FastScanComputer::kBlockSize;
            const size_t blk_count = std::min(FastScanComputer::kBlockSize, n_x - blk_start);
            TransposeBlock(x_codes, blk_start, blk_count, cached_transposed_.get() + blk * block_bytes);
        }
    }

    /// Lazily compute or_c_l2sqr_front[i] and or_c_l2sqr_mid[i] from float data.
    void EnsurePartialNormsCache(
        const float* x_float, size_t n_x, size_t d, uint32_t partial_d
    ) const {
        if (!pruning_partial_norms_dirty_ && cached_pruning_partial_d_ == partial_d) return;
        SKM_PROFILE_SCOPE("RaBitQ::EnsurePartialNormsCache");

        const size_t front_d = (partial_d / 8) * 8;
        const size_t mid_d = (d / 32) * 8;  // d/4, byte-aligned
        cached_or_c_l2sqr_front_.resize(n_x);
        cached_or_c_l2sqr_mid_.resize(n_x);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            const float* xi = x_float + i * d;
            float sum_front = 0, sum_mid = 0;
            const size_t min_fm = std::min(front_d, mid_d);
            const size_t max_fm = std::max(front_d, mid_d);
            size_t dim = 0;
            for (; dim < min_fm; ++dim) {
                const float diff = xi[dim] - centroid_[dim];
                sum_front += diff * diff;
            }
            sum_mid = sum_front;
            if (front_d <= mid_d) {
                // front_d < mid_d: continue accumulating into mid
                for (; dim < max_fm; ++dim) {
                    const float diff = xi[dim] - centroid_[dim];
                    sum_mid += diff * diff;
                }
            } else {
                // front_d > mid_d: continue accumulating into front
                for (; dim < max_fm; ++dim) {
                    const float diff = xi[dim] - centroid_[dim];
                    sum_front += diff * diff;
                }
            }
            cached_or_c_l2sqr_front_[i] = sum_front;
            cached_or_c_l2sqr_mid_[i] = sum_mid;
        }

        cached_pruning_partial_d_ = partial_d;
        pruning_partial_norms_dirty_ = false;
    }

    /// Extended LUT builder that also outputs partial centroid norms at front_d and mid_d.
    void QuantizeCentroidsAndBuildLUTsWithBounds(
        const float* y_float, size_t n_y, size_t d, size_t front_d, size_t mid_d,
        uint8_t* all_luts,
        float* c1, float* c2, float* c34, float* qr_to_c_l2sqr,
        float* qr_to_c_l2sqr_front, float* c34_front,
        float* qr_to_c_l2sqr_mid, float* c34_mid,
        uint8_t* centroid_planes
    ) const {
        SKM_PROFILE_SCOPE("RaBitQ::QuantizeCentroidsAndBuildLUTsWithBounds");
        const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const float max_val = static_cast<float>((1 << qb_) - 1);
        const size_t n_sub = 2 * binary_bytes_;
        const size_t front_d_clamped = std::min(front_d, d);
        const size_t mid_d_clamped = std::min(mid_d, d);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < n_y; ++j) {
            std::vector<float> rotated(d);
            float v_min = std::numeric_limits<float>::max();
            float v_max = std::numeric_limits<float>::lowest();
            float norm_sq = 0;
            float norm_sq_front = 0;
            float norm_sq_mid = 0;

            for (size_t dim = 0; dim < d; ++dim) {
                rotated[dim] = y_float[j * d + dim] - centroid_[dim];
                v_min = std::min(v_min, rotated[dim]);
                v_max = std::max(v_max, rotated[dim]);
                const float r2 = rotated[dim] * rotated[dim];
                norm_sq += r2;
                if (dim < front_d_clamped) norm_sq_front += r2;
                if (dim < mid_d_clamped) norm_sq_mid += r2;
            }
            qr_to_c_l2sqr[j] = norm_sq;
            qr_to_c_l2sqr_front[j] = norm_sq_front;
            qr_to_c_l2sqr_mid[j] = norm_sq_mid;

            float delta = (v_max - v_min) / max_val;
            if (delta < std::numeric_limits<float>::epsilon()) delta = 1.0f;
            const float inv_delta = 1.0f / delta;
            float sum_qq = 0;
            float sum_qq_front = 0;
            float sum_qq_mid = 0;

            std::vector<uint8_t> quantized(d);
            for (size_t dim = 0; dim < d; ++dim) {
                int v = static_cast<int>(std::lround((rotated[dim] - v_min) * inv_delta));
                v = std::max(0, std::min(v, static_cast<int>(max_val)));
                quantized[dim] = static_cast<uint8_t>(v);
                const float fv = static_cast<float>(v);
                sum_qq += fv;
                if (dim < front_d_clamped) sum_qq_front += fv;
                if (dim < mid_d_clamped) sum_qq_mid += fv;
            }

            c1[j] = 2.0f * delta * inv_sqrt_d;
            c2[j] = 2.0f * v_min * inv_sqrt_d;
            c34[j] = inv_sqrt_d * (delta * sum_qq + static_cast<float>(d) * v_min);
            c34_front[j] = inv_sqrt_d *
                (delta * sum_qq_front + static_cast<float>(front_d_clamped) * v_min);
            c34_mid[j] = inv_sqrt_d *
                (delta * sum_qq_mid + static_cast<float>(mid_d_clamped) * v_min);

            // Build LUTs (identical to QuantizeCentroidsAndBuildLUTs)
            uint8_t* lut_j = all_luts + j * n_sub * 16;
            for (size_t b = 0; b < binary_bytes_; ++b) {
                uint8_t* lut_lo = lut_j + (2 * b) * 16;
                uint8_t* lut_hi = lut_j + (2 * b + 1) * 16;
                uint8_t sq[8] = {0};
                for (int k = 0; k < 8 && (8 * b + k) < d; ++k) {
                    sq[k] = quantized[8 * b + k];
                }
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1) val += sq[0];
                    if (c & 2) val += sq[1];
                    if (c & 4) val += sq[2];
                    if (c & 8) val += sq[3];
                    lut_lo[c] = val;
                }
                for (int c = 0; c < 16; ++c) {
                    uint8_t val = 0;
                    if (c & 1) val += sq[4];
                    if (c & 2) val += sq[5];
                    if (c & 4) val += sq[6];
                    if (c & 8) val += sq[7];
                    lut_hi[c] = val;
                }
            }

            // Bit-transpose SQ values into qb planes (for popcount-based remaining distance)
            for (int b = 0; b < qb_; ++b) {
                uint8_t* plane = centroid_planes + (b * n_y + j) * binary_bytes_;
                std::memset(plane, 0, binary_bytes_);
                for (size_t dim = 0; dim < d; ++dim) {
                    if ((quantized[dim] >> b) & 1) {
                        plane[dim / 8] |= static_cast<uint8_t>(1 << (dim % 8));
                    }
                }
            }
        }
    }

    /// Compute full RaBitQ distance for a single (data, centroid) pair via LUT.
    static float ComputeFullDistanceViaLUT(
        const uint8_t* data_code,
        const uint8_t* lut_j,
        float c1j, float c2j, float c34j, float qr_j,
        uint32_t sum_q_i, float or_c_l2sqr_i, float dp_mult_i,
        size_t binary_bytes
    ) {
        uint16_t dot_qo = 0;
        for (size_t b = 0; b < binary_bytes; ++b) {
            const uint8_t byte = data_code[b];
            dot_qo += static_cast<uint16_t>(lut_j[(2 * b) * 16 + (byte & 0x0F)]) +
                      static_cast<uint16_t>(lut_j[(2 * b + 1) * 16 + (byte >> 4)]);
        }
        const float final_dot =
            c1j * static_cast<float>(dot_qo) +
            c2j * static_cast<float>(sum_q_i) - c34j;
        return or_c_l2sqr_i + qr_j - 2.0f * dp_mult_i * final_dot;
    }

    /// Overload that uses the instance's binary_bytes_.
    float ComputeFullDistanceViaLUT(
        const uint8_t* data_code,
        const uint8_t* lut_j,
        float c1j, float c2j, float c34j, float qr_j,
        uint32_t sum_q_i, float or_c_l2sqr_i, float dp_mult_i
    ) const {
        return ComputeFullDistanceViaLUT(
            data_code, lut_j, c1j, c2j, c34j, qr_j,
            sum_q_i, or_c_l2sqr_i, dp_mult_i, binary_bytes_
        );
    }

    /// Compute ADSampling ratio: (front_d/d) * (1 + eps0/sqrt(front_d))²
    static float ComputeADSamplingRatio(size_t front_d, size_t d) {
        if (front_d == 0 || front_d >= d) return 1.0f;
        const double eps0 = static_cast<double>(PRUNER_INITIAL_THRESHOLD);
        const double ratio =
            static_cast<double>(front_d) / static_cast<double>(d) *
            (1.0 + eps0 / std::sqrt(static_cast<double>(front_d))) *
            (1.0 + eps0 / std::sqrt(static_cast<double>(front_d)));
        return static_cast<float>(ratio);
    }

    // Pruning caches — checkpoint 1 (front, at partial_d)
    mutable std::vector<uint32_t> cached_sum_q_front_;      // [n_x] popcount of front bytes
    mutable std::vector<float> cached_or_c_l2sqr_front_;    // [n_x] partial norm over front dims
    mutable uint32_t cached_partial_d_ = 0;
    mutable uint32_t cached_pruning_partial_d_ = 0;
    mutable bool pruning_partial_norms_dirty_ = true;

    // Pruning caches — checkpoint 2 (mid, at d/4)
    mutable std::vector<uint32_t> cached_sum_q_mid_;        // [n_x] popcount of mid bytes
    mutable std::vector<float> cached_or_c_l2sqr_mid_;      // [n_x] partial norm over mid dims
};

} // namespace skmeans

// #endif // HAS_FAISS
