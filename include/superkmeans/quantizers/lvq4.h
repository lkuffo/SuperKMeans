#pragma once

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <vector>

#include <numkong/numkong.h>

namespace skmeans {

/**
 * @brief LVQ4 (Locally-adaptive Vector Quantization, 4-bit) quantizer.
 *
 * Per-vector adaptive scalar quantization: each vector stores its own
 * scale and bias, capturing local dynamic range. Encoding:
 *   code[dim] = round((x[dim] - bias) / scale), clamped to [0,15]
 * where bias = min(x), scale = (max(x) - min(x)) / 15.
 *
 * Code layout per vector:
 *   [d/2 packed u4x2 bytes] [float scale: 4B] [float bias: 4B]
 *   CodeSize(d) = d/2 + 8
 *
 * Distance formula between LVQ4 vectors x_i, y_j:
 *   L2(x_i, y_j) = s_i*Scx + s_j*Scy - 2*s_i*s_j*<cx,cy>
 *                    + 2*db*(s_i*Scx - s_j*Scy) + d*db
 * where db = b_i - b_j, and <cx,cy> is the integer dot product of nibbles.
 *
 * Uses NumKong u4 GEMM (nk_dots_packed_u4) for bulk dot products and
 * u4 Horizontal SIMD kernels for per-pair distance in pruning survivors.
 */
class LVQ4Quantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;
    using u4_computer = DistanceComputer<DistanceFunction::l2, Quantization::u4>;

    void Fit(const float* /*data*/, size_t /*n*/, size_t d) override {
        SKM_PROFILE_SCOPE("LVQ4::Fit");
        assert(d % 2 == 0 && "LVQ4 requires even dimensionality");
        d_ = d;
        nibble_bytes_ = d / 2;
        code_size_ = nibble_bytes_ + 8;
        dots_buf_.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        packed_buf_.resize(nk_dots_packed_size_u4(Y_BATCH_SIZE, d));
        fitted_ = true;
    }

    void Encode(const float* in, quantized_t* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("LVQ4::Encode");
        assert(fitted_ && d == d_);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const float* x = in + i * d;
            uint8_t* code = out + i * code_size_;

            float v_min = x[0], v_max = x[0];
            for (size_t dim = 1; dim < d; ++dim) {
                v_min = std::min(v_min, x[dim]);
                v_max = std::max(v_max, x[dim]);
            }

            float range = v_max - v_min;
            if (range < 1e-30f) range = 1e-30f;
            float scale = range / 15.0f;
            float inv_scale = 1.0f / scale;
            float bias = v_min;

            for (size_t dim = 0; dim < d; dim += 2) {
                int q_lo = static_cast<int>(std::lround((x[dim] - bias) * inv_scale));
                int q_hi = static_cast<int>(std::lround((x[dim + 1] - bias) * inv_scale));
                q_lo = std::max(0, std::min(15, q_lo));
                q_hi = std::max(0, std::min(15, q_hi));
                code[dim / 2] = static_cast<uint8_t>(q_lo | (q_hi << 4));
            }

            float* footer = (float*)(code + nibble_bytes_);
            footer[0] = scale;
            footer[1] = bias;
        }
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("LVQ4::Decode");
        assert(fitted_ && d == d_);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = in + i * code_size_;
            float* dst = out + i * d;

            const float* footer = (const float*)(code + nibble_bytes_);
            float scale = footer[0];
            float bias = footer[1];

            for (size_t b = 0; b < nibble_bytes_; ++b) {
                dst[2 * b]     = scale * static_cast<float>(code[b] & 0x0F) + bias;
                dst[2 * b + 1] = scale * static_cast<float>(code[b] >> 4)   + bias;
            }
        }
    }

    void ComputeNorms(
        const quantized_t* data, size_t n, size_t d, float* out_norms
    ) const override {
        SKM_PROFILE_SCOPE("LVQ4::ComputeNorms");
        assert(fitted_ && d == d_);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = data + i * code_size_;
            const float* footer = (const float*)(code + nibble_bytes_);
            float s = footer[0];
            float b = footer[1];

            uint32_t sum_c = 0, sum_c_sq = 0;
            for (size_t byte = 0; byte < nibble_bytes_; ++byte) {
                uint8_t lo = code[byte] & 0x0F;
                uint8_t hi = code[byte] >> 4;
                sum_c += lo + hi;
                sum_c_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
            }
            out_norms[i] = s * s * static_cast<float>(sum_c_sq)
                         + 2.0f * s * b * static_cast<float>(sum_c)
                         + static_cast<float>(d) * b * b;
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
        SKM_PROFILE_SCOPE("LVQ4::FindNearestNeighbor");
        assert(fitted_);

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);
        const uint8_t* y_codes = reinterpret_cast<const uint8_t*>(y);

        EnsureCodeFactorsCache(x_codes, n_x);

        CentroidFactors cf;
        ExtractCentroidFactors(y_codes, n_y, 0, 0, cf);

        std::fill_n(out_distances, n_x, std::numeric_limits<float>::max());
        std::fill_n(out_knn, n_x, 0u);

        const auto* x_u4 = (const nk_u4x2_t*)x_codes;
        const auto* y_u4 = (const nk_u4x2_t*)y_codes;

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("LVQ4::FindNearestNeighbor/gemm");
                    // Pack centroid batch for NumKong
                    const size_t pack_size = nk_dots_packed_size_u4(batch_n_y, d);
                    if (pack_size > packed_buf_.size()) packed_buf_.resize(pack_size);
                    nk_dots_pack_u4(
                        y_u4 + j * code_size_, batch_n_y, d, code_size_, packed_buf_.data()
                    );

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
                            nk_dots_packed_u4(
                                x_u4 + (i + start) * code_size_,
                                packed_buf_.data(),
                                dots_buf_.data() + start * batch_n_y,
                                count,
                                batch_n_y,
                                d,
                                code_size_,
                                c_stride
                            );
                        }
                    }
                }

                // Apply LVQ4 distance formula per pair
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const size_t idx = i + r;
                    const uint32_t* dots_row = dots_buf_.data() + r * batch_n_y;
                    const float si = cached_scales_[idx];
                    const float bi = cached_biases_[idx];
                    const float norm_x_i = cached_norm_x_full_[idx];
                    const float two_si = 2.0f * si;
                    const float two_A_x_i = 2.0f * cached_A_x_full_[idx];
                    const float two_bi = 2.0f * bi;

                    for (size_t c = 0; c < batch_n_y; ++c) {
                        const size_t j_idx = j + c;
                        float dist = norm_x_i + cf.norm_y_full[j_idx]
                                   - two_si * cf.scales[j_idx] * static_cast<float>(dots_row[c])
                                   - two_A_x_i * cf.biases[j_idx]
                                   - two_bi * cf.sj_sum_cy_full[j_idx];

                        if (dist < out_distances[idx]) {
                            out_distances[idx] = dist;
                            out_knn[idx] = static_cast<uint32_t>(j_idx);
                        }
                    }
                }
            }
        }
    }

    void CacheDataPartialNorms(
        const quantized_t* data, size_t n, size_t /*d*/, uint32_t partial_d
    ) override {
        SKM_PROFILE_SCOPE("LVQ4::CacheDataPartialNorms");
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(data);
        const size_t front_bytes = partial_d / 2;
        const size_t mid_bytes = d_ / 8;  // d/4, in packed bytes
        const float front_d_f = static_cast<float>(front_bytes * 2);
        const float mid_d_f = static_cast<float>(mid_bytes * 2);

        cached_sum_cx_front_.resize(n);
        cached_sum_cx_sq_front_.resize(n);
        cached_sum_cx_mid_.resize(n);
        cached_sum_cx_sq_mid_.resize(n);
        cached_norm_x_front_.resize(n);
        cached_A_x_front_.resize(n);
        cached_norm_x_mid_.resize(n);
        cached_A_x_mid_.resize(n);
        cached_partial_d_ = partial_d;

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes + i * code_size_;
            const float* footer = (const float*)(code + nibble_bytes_);
            const float si = footer[0];
            const float bi = footer[1];

            uint32_t sum = 0, sum_sq = 0;
            size_t b = 0;
            const size_t min_fb = std::min(front_bytes, mid_bytes);
            const size_t max_fb = std::max(front_bytes, mid_bytes);

            for (; b < min_fb; ++b) {
                uint8_t lo = code[b] & 0x0F;
                uint8_t hi = code[b] >> 4;
                sum += lo + hi;
                sum_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
            }

            if (front_bytes <= mid_bytes) {
                cached_sum_cx_front_[i] = sum;
                cached_sum_cx_sq_front_[i] = sum_sq;
                float sf = static_cast<float>(sum), sqf = static_cast<float>(sum_sq);
                cached_norm_x_front_[i] = si * si * sqf + 2.0f * bi * si * sf + front_d_f * bi * bi;
                cached_A_x_front_[i] = si * sf + front_d_f * bi;
                for (; b < max_fb; ++b) {
                    uint8_t lo = code[b] & 0x0F;
                    uint8_t hi = code[b] >> 4;
                    sum += lo + hi;
                    sum_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
                }
                cached_sum_cx_mid_[i] = sum;
                cached_sum_cx_sq_mid_[i] = sum_sq;
                sf = static_cast<float>(sum); sqf = static_cast<float>(sum_sq);
                cached_norm_x_mid_[i] = si * si * sqf + 2.0f * bi * si * sf + mid_d_f * bi * bi;
                cached_A_x_mid_[i] = si * sf + mid_d_f * bi;
            } else {
                cached_sum_cx_mid_[i] = sum;
                cached_sum_cx_sq_mid_[i] = sum_sq;
                float sf = static_cast<float>(sum), sqf = static_cast<float>(sum_sq);
                cached_norm_x_mid_[i] = si * si * sqf + 2.0f * bi * si * sf + mid_d_f * bi * bi;
                cached_A_x_mid_[i] = si * sf + mid_d_f * bi;
                for (; b < max_fb; ++b) {
                    uint8_t lo = code[b] & 0x0F;
                    uint8_t hi = code[b] >> 4;
                    sum += lo + hi;
                    sum_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
                }
                cached_sum_cx_front_[i] = sum;
                cached_sum_cx_sq_front_[i] = sum_sq;
                sf = static_cast<float>(sum); sqf = static_cast<float>(sum_sq);
                cached_norm_x_front_[i] = si * si * sqf + 2.0f * bi * si * sf + front_d_f * bi * bi;
                cached_A_x_front_[i] = si * sf + front_d_f * bi;
            }
        }
    }

    void CacheCentroidPartialNorms(
        const quantized_t* /*centroids*/, size_t /*n*/, size_t /*d*/, uint32_t /*partial_d*/
    ) override {
        // No-op: centroid partials computed in ExtractCentroidFactors.
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
        uint32_t partial_d,
        size_t* out_not_pruned_counts
    ) const override {
        SKM_PROFILE_SCOPE("LVQ4::FindNearestNeighborWithPruning");
        assert(fitted_);

        const uint8_t* x_codes = reinterpret_cast<const uint8_t*>(x);
        const uint8_t* y_codes = reinterpret_cast<const uint8_t*>(y);

        EnsureCodeFactorsCache(x_codes, n_x);

        // Pruning geometry
        const size_t front_bytes = partial_d / 2;
        const size_t front_d = front_bytes * 2;
        const size_t mid_bytes = d / 8;  // d/4, in packed bytes
        const size_t mid_d = mid_bytes * 2;
        const bool use_mid = (front_bytes < mid_bytes) && (mid_bytes < nibble_bytes_);
        const size_t phase3_start = use_mid ? mid_bytes : front_bytes;

        // ADSampling ratios
        const float ad_ratio_front = ComputeADSamplingRatio(front_d, d);
        const float ad_ratio_mid = use_mid ? ComputeADSamplingRatio(mid_d, d) : 1.0f;

        CentroidFactors cf;
        ExtractCentroidFactors(y_codes, n_y, front_d, mid_d, cf);

        const auto* x_u4 = (const nk_u4x2_t*)x_codes;
        const auto* y_u4 = (const nk_u4x2_t*)y_codes;

        for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
            const size_t batch_n_x = std::min(X_BATCH_SIZE, n_x - i);

            for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
                const size_t batch_n_y = std::min(Y_BATCH_SIZE, n_y - j);

                {
                    SKM_PROFILE_SCOPE("LVQ4::FindNearestNeighborWithPruning/front_gemm");
                    // Pack centroid batch for partial_d GEMM
                    const size_t pack_size = nk_dots_packed_size_u4(batch_n_y, partial_d);
                    if (pack_size > packed_buf_.size()) packed_buf_.resize(pack_size);
                    nk_dots_pack_u4(
                        y_u4 + j * code_size_, batch_n_y, partial_d, code_size_,
                        packed_buf_.data()
                    );

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
                            nk_dots_packed_u4(
                                x_u4 + (i + start) * code_size_,
                                packed_buf_.data(),
                                dots_buf_.data() + start * batch_n_y,
                                count,
                                batch_n_y,
                                partial_d,
                                code_size_,
                                c_stride
                            );
                        }
                    }
                }

                {
                    SKM_PROFILE_SCOPE("LVQ4::FindNearestNeighborWithPruning/prune_and_resolve");
#if defined(__clang__)
#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
#else
#pragma omp parallel for num_threads(g_n_threads)
#endif
                    for (size_t r = 0; r < batch_n_x; ++r) {
                        const size_t i_idx = i + r;
                        const uint32_t* dots_row = dots_buf_.data() + r * batch_n_y;
                        const uint8_t* x_code = x_codes + i_idx * code_size_;

                        const float si = cached_scales_[i_idx];
                        const float bi = cached_biases_[i_idx];
                        const float two_si = 2.0f * si;
                        const float two_bi = 2.0f * bi;
                        const float norm_x_front_i = cached_norm_x_front_[i_idx];
                        const float two_A_x_front_i = 2.0f * cached_A_x_front_[i_idx];
                        const float norm_x_mid_i = cached_norm_x_mid_[i_idx];
                        const float two_A_x_mid_i = 2.0f * cached_A_x_mid_[i_idx];
                        const float norm_x_full_i = cached_norm_x_full_[i_idx];
                        const float two_A_x_full_i = 2.0f * cached_A_x_full_[i_idx];

                        // Phase 1: threshold from previous assignment
                        float best_dist;
                        uint32_t best_idx;
                        if (j == 0) {
                            const uint32_t prev_j = out_knn[i_idx];
                            best_idx = prev_j;
                            best_dist = ComputeFullDistance(
                                x_code, y_codes + prev_j * code_size_,
                                si, bi,
                                cached_sum_cx_sq_[i_idx],
                                norm_x_full_i, cached_A_x_full_[i_idx],
                                cf.scales[prev_j], cf.biases[prev_j],
                                cf.sum_cy_sq[prev_j],
                                cf.norm_y_full[prev_j], cf.sj_sum_cy_full[prev_j]
                            );
                            out_not_pruned_counts[i_idx] = 0;
                        } else {
                            best_dist = out_distances[i_idx];
                            best_idx = out_knn[i_idx];
                        }

                        for (size_t c = 0; c < batch_n_y; ++c) {
                            const size_t j_idx = j + c;
                            const float sj = cf.scales[j_idx];
                            const float bj = cf.biases[j_idx];

                            // Front checkpoint: optimized partial LVQ4 distance
                            float partial_l2 = norm_x_front_i + cf.norm_y_front[j_idx]
                                - two_si * sj * static_cast<float>(dots_row[c])
                                - two_A_x_front_i * bj
                                - two_bi * cf.sj_sum_cy_front_f[j_idx];

                            if (partial_l2 > best_dist * ad_ratio_front)
                                continue;

                            out_not_pruned_counts[i_idx]++;
                            uint32_t dot_accumulated = dots_row[c];

                            const uint8_t* y_code = y_codes + j_idx * code_size_;

                            // Mid checkpoint: u4 Horizontal on [front_bytes, mid_bytes)
                            if (use_mid) {
                                uint32_t gap_l2_int = u4_computer::Horizontal(
                                    (const nk_u4x2_t*)(x_code + front_bytes),
                                    (const nk_u4x2_t*)(y_code + front_bytes),
                                    mid_bytes - front_bytes
                                );
                                uint32_t sum_cx_sq_gap =
                                    cached_sum_cx_sq_mid_[i_idx] - cached_sum_cx_sq_front_[i_idx];
                                uint32_t sum_cy_sq_gap =
                                    cf.sum_cy_sq_mid[j_idx] - cf.sum_cy_sq_front[j_idx];
                                uint32_t gap_dot = (sum_cx_sq_gap + sum_cy_sq_gap - gap_l2_int) / 2;
                                dot_accumulated += gap_dot;

                                float mid_l2 = norm_x_mid_i + cf.norm_y_mid[j_idx]
                                    - two_si * sj * static_cast<float>(dot_accumulated)
                                    - two_A_x_mid_i * bj
                                    - two_bi * cf.sj_sum_cy_mid_f[j_idx];

                                if (mid_l2 > best_dist * ad_ratio_mid)
                                    continue;
                            }

                            // Rest: u4 Horizontal on [phase3_start, nibble_bytes)
                            uint32_t rest_l2_int = u4_computer::Horizontal(
                                (const nk_u4x2_t*)(x_code + phase3_start),
                                (const nk_u4x2_t*)(y_code + phase3_start),
                                nibble_bytes_ - phase3_start
                            );
                            uint32_t sum_cx_sq_at_start = use_mid
                                ? cached_sum_cx_sq_mid_[i_idx]
                                : cached_sum_cx_sq_front_[i_idx];
                            uint32_t sum_cy_sq_at_start = use_mid
                                ? cf.sum_cy_sq_mid[j_idx]
                                : cf.sum_cy_sq_front[j_idx];
                            uint32_t sum_cx_sq_rest =
                                cached_sum_cx_sq_[i_idx] - sum_cx_sq_at_start;
                            uint32_t sum_cy_sq_rest =
                                cf.sum_cy_sq[j_idx] - sum_cy_sq_at_start;
                            uint32_t rest_dot =
                                (sum_cx_sq_rest + sum_cy_sq_rest - rest_l2_int) / 2;
                            uint32_t full_dot = dot_accumulated + rest_dot;

                            float full_l2 = norm_x_full_i + cf.norm_y_full[j_idx]
                                - two_si * sj * static_cast<float>(full_dot)
                                - two_A_x_full_i * bj
                                - two_bi * cf.sj_sum_cy_full[j_idx];

                            if (full_l2 < best_dist) {
                                best_dist = full_l2;
                                best_idx = static_cast<uint32_t>(j_idx);
                            }
                        }

                        out_distances[i_idx] = best_dist;
                        out_knn[i_idx] = best_idx;
                    }
                }
            }
        }
    }


    size_t CodeSize(size_t d) const override { return d / 2 + 8; }
    bool IsFitted() const override { return fitted_; }
    bool SupportsPruning() const override { return true; }
    bool NeedsPDXLayout() const override { return false; }


  private:
    size_t d_ = 0;
    size_t nibble_bytes_ = 0;      // d/2
    size_t code_size_ = 0;         // d/2 + 8
    bool fitted_ = false;

    // static constexpr size_t X_BATCH_SIZE = 65536;
    // static constexpr size_t Y_BATCH_SIZE = 256;

    // Per-data caches (keyed by pointer + count for invalidation)
    mutable const uint8_t* cached_x_ptr_ = nullptr;
    mutable size_t cached_n_x_ = 0;
    mutable std::vector<float> cached_scales_, cached_biases_;
    mutable std::vector<uint32_t> cached_sum_cx_, cached_sum_cx_sq_;

    // GEMM buffers
    mutable std::vector<uint32_t> dots_buf_;
    mutable std::vector<char> packed_buf_;

    // Pruning checkpoint caches
    mutable std::vector<uint32_t> cached_sum_cx_front_, cached_sum_cx_sq_front_;
    mutable std::vector<uint32_t> cached_sum_cx_mid_, cached_sum_cx_sq_mid_;
    mutable uint32_t cached_partial_d_ = 0;

    // Precomputed float caches: norm_x = si²·Σcx² + 2·bi·si·Σcx + D·bi²
    //                           A_x    = si·Σcx + D·bi
    mutable std::vector<float> cached_norm_x_full_, cached_A_x_full_;
    mutable std::vector<float> cached_norm_x_front_, cached_A_x_front_;
    mutable std::vector<float> cached_norm_x_mid_, cached_A_x_mid_;

    struct CentroidFactors {
        std::vector<float> scales, biases;
        std::vector<uint32_t> sum_cy, sum_cy_sq;
        std::vector<uint32_t> sum_cy_front, sum_cy_sq_front;
        std::vector<uint32_t> sum_cy_mid, sum_cy_sq_mid;
        // Precomputed: norm_y = sj²·Σcy² + 2·bj·sj·Σcy + D·bj²
        //              sj_sum_cy = sj·Σcy
        std::vector<float> norm_y_full, sj_sum_cy_full;
        std::vector<float> norm_y_front, sj_sum_cy_front_f;
        std::vector<float> norm_y_mid, sj_sum_cy_mid_f;
    };

    void EnsureCodeFactorsCache(const uint8_t* x_codes, size_t n_x) const {
        if (cached_x_ptr_ == x_codes && cached_n_x_ == n_x) return;
        SKM_PROFILE_SCOPE("LVQ4::EnsureCodeFactorsCache");

        cached_scales_.resize(n_x);
        cached_biases_.resize(n_x);
        cached_sum_cx_.resize(n_x);
        cached_sum_cx_sq_.resize(n_x);
        cached_norm_x_full_.resize(n_x);
        cached_A_x_full_.resize(n_x);

        const float d_f = static_cast<float>(d_);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n_x; ++i) {
            const uint8_t* code = x_codes + i * code_size_;
            const float* footer = (const float*)(code + nibble_bytes_);
            const float si = footer[0];
            const float bi = footer[1];
            cached_scales_[i] = si;
            cached_biases_[i] = bi;

            uint32_t sum = 0, sum_sq = 0;
            for (size_t b = 0; b < nibble_bytes_; ++b) {
                uint8_t lo = code[b] & 0x0F;
                uint8_t hi = code[b] >> 4;
                sum += lo + hi;
                sum_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
            }
            cached_sum_cx_[i] = sum;
            cached_sum_cx_sq_[i] = sum_sq;

            const float sum_f = static_cast<float>(sum);
            const float sum_sq_f = static_cast<float>(sum_sq);
            cached_norm_x_full_[i] = si * si * sum_sq_f + 2.0f * bi * si * sum_f + d_f * bi * bi;
            cached_A_x_full_[i] = si * sum_f + d_f * bi;
        }

        cached_x_ptr_ = x_codes;
        cached_n_x_ = n_x;
    }

    void ExtractCentroidFactors(
        const uint8_t* y_codes, size_t n_y,
        size_t front_d, size_t mid_d,
        CentroidFactors& cf
    ) const {
        SKM_PROFILE_SCOPE("LVQ4::ExtractCentroidFactors");
        const size_t front_bytes = front_d / 2;
        const size_t mid_bytes = mid_d / 2;

        cf.scales.resize(n_y);
        cf.biases.resize(n_y);
        cf.sum_cy.resize(n_y);
        cf.sum_cy_sq.resize(n_y);
        cf.sum_cy_front.resize(n_y);
        cf.sum_cy_sq_front.resize(n_y);
        cf.sum_cy_mid.resize(n_y);
        cf.sum_cy_sq_mid.resize(n_y);
        cf.norm_y_full.resize(n_y);
        cf.sj_sum_cy_full.resize(n_y);
        cf.norm_y_front.resize(n_y);
        cf.sj_sum_cy_front_f.resize(n_y);
        cf.norm_y_mid.resize(n_y);
        cf.sj_sum_cy_mid_f.resize(n_y);

        const float d_f = static_cast<float>(d_);
        const float front_d_f = static_cast<float>(front_d);
        const float mid_d_f = static_cast<float>(mid_d);

#pragma omp parallel for num_threads(g_n_threads)
        for (size_t j = 0; j < n_y; ++j) {
            const uint8_t* code_j = y_codes + j * code_size_;
            const float* footer = (const float*)(code_j + nibble_bytes_);
            cf.scales[j] = footer[0];
            cf.biases[j] = footer[1];

            uint32_t sum = 0, sum_sq = 0;
            bool saved_front = (front_bytes == 0);
            bool saved_mid = (mid_bytes == 0);

            for (size_t b = 0; b < nibble_bytes_; ++b) {
                uint8_t lo = code_j[b] & 0x0F;
                uint8_t hi = code_j[b] >> 4;
                sum += lo + hi;
                sum_sq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;

                if (!saved_front && b + 1 == front_bytes) {
                    cf.sum_cy_front[j] = sum;
                    cf.sum_cy_sq_front[j] = sum_sq;
                    saved_front = true;
                }
                if (!saved_mid && b + 1 == mid_bytes) {
                    cf.sum_cy_mid[j] = sum;
                    cf.sum_cy_sq_mid[j] = sum_sq;
                    saved_mid = true;
                }
            }

            // Fallback: if checkpoint boundary wasn't exactly hit, recompute
            if (!saved_front) {
                uint32_t s = 0, ssq = 0;
                for (size_t b = 0; b < std::min(front_bytes, nibble_bytes_); ++b) {
                    uint8_t lo = code_j[b] & 0x0F;
                    uint8_t hi = code_j[b] >> 4;
                    s += lo + hi;
                    ssq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
                }
                cf.sum_cy_front[j] = s;
                cf.sum_cy_sq_front[j] = ssq;
            }
            if (!saved_mid) {
                uint32_t s = 0, ssq = 0;
                for (size_t b = 0; b < std::min(mid_bytes, nibble_bytes_); ++b) {
                    uint8_t lo = code_j[b] & 0x0F;
                    uint8_t hi = code_j[b] >> 4;
                    s += lo + hi;
                    ssq += static_cast<uint32_t>(lo) * lo + static_cast<uint32_t>(hi) * hi;
                }
                cf.sum_cy_mid[j] = s;
                cf.sum_cy_sq_mid[j] = ssq;
            }

            cf.sum_cy[j] = sum;
            cf.sum_cy_sq[j] = sum_sq;

            // Precompute float values for optimized distance formula
            const float sj = cf.scales[j];
            const float bj = cf.biases[j];

            float sc_f = static_cast<float>(cf.sum_cy[j]);
            float scsq_f = static_cast<float>(cf.sum_cy_sq[j]);
            cf.norm_y_full[j] = sj * sj * scsq_f + 2.0f * bj * sj * sc_f + d_f * bj * bj;
            cf.sj_sum_cy_full[j] = sj * sc_f;

            sc_f = static_cast<float>(cf.sum_cy_front[j]);
            scsq_f = static_cast<float>(cf.sum_cy_sq_front[j]);
            cf.norm_y_front[j] = sj * sj * scsq_f + 2.0f * bj * sj * sc_f + front_d_f * bj * bj;
            cf.sj_sum_cy_front_f[j] = sj * sc_f;

            sc_f = static_cast<float>(cf.sum_cy_mid[j]);
            scsq_f = static_cast<float>(cf.sum_cy_sq_mid[j]);
            cf.norm_y_mid[j] = sj * sj * scsq_f + 2.0f * bj * sj * sc_f + mid_d_f * bj * bj;
            cf.sj_sum_cy_mid_f[j] = sj * sc_f;
        }
    }

    float ComputeFullDistance(
        const uint8_t* x_code,
        const uint8_t* y_code,
        float si, float bi,
        uint32_t sum_cx_sq,
        float norm_x_full, float A_x_full,
        float sj, float bj,
        uint32_t sum_cy_sq,
        float norm_y_full, float sj_sum_cy_full
    ) const {
        uint32_t l2_int = u4_computer::Horizontal(
            (const nk_u4x2_t*)x_code,
            (const nk_u4x2_t*)y_code,
            nibble_bytes_
        );
        uint32_t int_dot = (sum_cx_sq + sum_cy_sq - l2_int) / 2;
        return norm_x_full + norm_y_full
             - 2.0f * si * sj * static_cast<float>(int_dot)
             - 2.0f * A_x_full * bj
             - 2.0f * bi * sj_sum_cy_full;
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

};

} // namespace skmeans
