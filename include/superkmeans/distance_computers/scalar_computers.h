#pragma once

#include <cassert>
#include <cstring>

#include "superkmeans/common.h"

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class ScalarComputer {};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::u8> {

  public:
    using distance_t = pdx_distance_t<Quantization::u8>;
    using data_t = skmeans_value_t<Quantization::u8>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        distance_t distance = 0;
        SKM_VECTORIZE_LOOP
        for (size_t i = 0; i < num_dimensions; ++i) {
            int diff = static_cast<int>(vector1[i]) - static_cast<int>(vector2[i]);
            distance += diff * diff;
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t<Quantization::f32>;
    using data_t = skmeans_value_t<Quantization::f32>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        distance_t distance = 0.0;
        SKM_VECTORIZE_LOOP
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance_t to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::dp, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t<Quantization::f32>;
    using data_t = skmeans_value_t<Quantization::f32>;

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        distance_t distance = 0.0;
        SKM_VECTORIZE_LOOP
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance += vector1[dimension_idx] * vector2[dimension_idx];
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::u4> {
  public:
    using distance_t = pdx_distance_t<Quantization::u4>;
    using data_t = skmeans_value_t<Quantization::u4>;

    /**
     * @brief Computes L2² distance between two packed u4x2 vectors (scalar).
     * Adapted from nk_sqeuclidean_u4_serial in NumKong.
     * @param vector1 Packed u4x2 input vector 1
     * @param vector2 Packed u4x2 input vector 2
     * @param num_packed_bytes Number of packed bytes to process (each byte = 2 dims)
     * @return L2² distance as uint32_t
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_packed_bytes
    ) {
        distance_t distance = 0;
        for (size_t i = 0; i < num_packed_bytes; ++i) {
            int32_t a_lo = vector1[i] & 0x0F;
            int32_t b_lo = vector2[i] & 0x0F;
            int32_t a_hi = (vector1[i] >> 4) & 0x0F;
            int32_t b_hi = (vector2[i] >> 4) & 0x0F;
            int32_t diff_lo = a_lo - b_lo;
            int32_t diff_hi = a_hi - b_hi;
            distance += static_cast<uint32_t>(diff_lo * diff_lo + diff_hi * diff_hi);
        }
        return distance;
    };
};

template <>
class ScalarComputer<DistanceFunction::l2, Quantization::b8> {
  public:
    using distance_t = pdx_distance_t<Quantization::b8>;
    using data_t = skmeans_value_t<Quantization::b8>;

    /**
     * @brief Computes popcount(a AND b) — binary inner product.
     * @param vector1 Binary vector 1 (packed bytes)
     * @param vector2 Binary vector 2 (packed bytes)
     * @param num_bytes Number of bytes to process (d/8 for d-bit vectors)
     * @return popcount of bitwise AND
     */
    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_bytes
    ) {
        uint32_t count = 0;
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(vector1);
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(vector2);
        size_t n_words = num_bytes / 8;
        for (size_t i = 0; i < n_words; ++i) {
            count += static_cast<uint32_t>(__builtin_popcountll(a64[i] & b64[i]));
        }
        for (size_t i = n_words * 8; i < num_bytes; ++i) {
            count += static_cast<uint32_t>(__builtin_popcount(vector1[i] & vector2[i]));
        }
        return count;
    };

    /**
     * @brief Fused multi-bitplane popcount: sum over bp of (popcount(data AND plane_bp) << bp).
     * Loads data once and reuses across all qb bitplanes.
     * @param data Binary data vector (packed bytes)
     * @param planes_base Pointer to bitplane 0 (stride plane_stride to next)
     * @param plane_stride Byte offset between consecutive bitplanes
     * @param num_bytes Number of bytes per vector
     * @param qb Number of bitplanes
     */
    static uint32_t HorizontalMultiPlane(
        const data_t* SKM_RESTRICT data,
        const data_t* planes_base,
        size_t plane_stride,
        size_t num_bytes,
        int qb
    ) {
        uint32_t result = 0;
        const uint64_t* d64 = reinterpret_cast<const uint64_t*>(data);
        size_t n_words = num_bytes / 8;
        for (size_t i = 0; i < n_words; ++i) {
            uint64_t x = d64[i];
            for (int bp = 0; bp < qb; ++bp) {
                const uint64_t* p64 = reinterpret_cast<const uint64_t*>(
                    planes_base + bp * plane_stride);
                result += static_cast<uint32_t>(__builtin_popcountll(x & p64[i])) << bp;
            }
        }
        for (size_t i = n_words * 8; i < num_bytes; ++i) {
            for (int bp = 0; bp < qb; ++bp) {
                result += static_cast<uint32_t>(
                    __builtin_popcount(data[i] & (planes_base + bp * plane_stride)[i])) << bp;
            }
        }
        return result;
    }
};

template <Quantization q>
class ScalarUtilsComputer {
  public:
    using data_t = skmeans_value_t<q>;
    using pdx_dist_t = pdx_distance_t<q>;

    static void FlipSign(const data_t*, data_t*, const uint32_t*, size_t) {
        assert(false && "FlipSign not supported");
    }

    static void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        pdx_dist_t pruning_threshold,
        const pdx_dist_t* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        for (size_t i = 0; i < n_vectors; ++i) {
            pruning_positions[n_vectors_not_pruned] = i;
            n_vectors_not_pruned += pruning_distances[i] < pruning_threshold;
        }
    }

    static void PackU8ToU4x2(const uint8_t*, uint8_t*, size_t) {
        assert(false && "PackU8ToU4x2 not applicable");
    }
};

template <>
class ScalarUtilsComputer<Quantization::f32> {
  public:
    using data_t = skmeans_value_t<Quantization::f32>;

    /**
     * @brief Flip sign of floats based on a mask (single vector).
     * @param data Input vector (d elements)
     * @param out Output vector (can be same as data for in-place)
     * @param masks Bitmask array (0x80000000 to flip, 0 to keep)
     * @param d Number of dimensions
     */
    static void FlipSign(const data_t* data, data_t* out, const uint32_t* masks, size_t d) {
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        SKM_VECTORIZE_LOOP
        for (size_t j = 0; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }

    /**
     * @brief Initializes positions array with indices of non-pruned vectors (scalar fallback).
     *
     * @param n_vectors Number of vectors to process
     * @param n_vectors_not_pruned Output: count of vectors passing threshold (updated)
     * @param pruning_positions Output array of indices that passed (compacted)
     * @param pruning_threshold Threshold value for comparison
     * @param pruning_distances Input array of distances to compare
     */
    static void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        data_t pruning_threshold,
        const data_t* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    static void PackU8ToU4x2(const uint8_t*, uint8_t*, size_t) {
        assert(false && "PackU8ToU4x2 not applicable for f32");
    }
};

template <>
class ScalarUtilsComputer<Quantization::u4> {
  public:
    using data_t = skmeans_value_t<Quantization::u4>;
    using pdx_dist_t = pdx_distance_t<Quantization::u4>;

    static void FlipSign(const data_t*, data_t*, const uint32_t*, size_t) {
        assert(false && "FlipSign not supported for u4");
    }

    static void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        pdx_dist_t pruning_threshold,
        const pdx_dist_t* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        for (size_t i = 0; i < n_vectors; ++i) {
            pruning_positions[n_vectors_not_pruned] = i;
            n_vectors_not_pruned += pruning_distances[i] < pruning_threshold;
        }
    }

    /**
     * @brief Pack u8 values [0,15] into u4x2 format (two nibbles per byte).
     *
     * dst[k] = (src[2k] & 0x0F) | ((src[2k+1] & 0x0F) << 4)
     *
     * @param src Input u8 array (count elements, each in [0,15])
     * @param dst Output u4x2 array (count/2 bytes)
     * @param count Number of input u8 elements (must be even)
     */
    static void PackU8ToU4x2(const uint8_t* src, uint8_t* dst, size_t count) {
        assert(count % 2 == 0);
        const size_t n_packed = count / 2;
        SKM_VECTORIZE_LOOP
        for (size_t k = 0; k < n_packed; ++k) {
            dst[k] = (src[2 * k] & 0x0F) | ((src[2 * k + 1] & 0x0F) << 4);
        }
    }
};

class ScalarFastScanComputer {
  public:
    static constexpr size_t kBlockSize = 32;

    /**
     * @brief Compact surviving positions where partial_l2[k] <= threshold[k].
     *
     * Caller precomputes threshold[k] = best_dist[k] * adsampling_ratio.
     */
    static void RabitQCompactSurvivors(
        size_t n_vectors,
        size_t& n_survivors,
        uint32_t* survivor_positions,
        const float* partial_l2,
        const float* threshold
    ) {
        n_survivors = 0;
        for (size_t k = 0; k < n_vectors; ++k) {
            survivor_positions[n_survivors] = static_cast<uint32_t>(k);
            n_survivors += partial_l2[k] <= threshold[k];
        }
    }

    /**
     * @brief Compute RaBitQ partial L2 distances for a block of points against one centroid.
     *
     * For each point k in [0, blk_count):
     *   fdt = c1j * float(partial_dot[k]) + c2j * float(sum_q[k]) - c34j
     *   out[k] = or_c_l2sqr[k] + qr_j - 2 * dp_mult[k] * fdt
     *
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     */
    /**
     * @brief Compute RaBitQ partial L2 distances for a block of points against one centroid.
     *
     * @tparam U32Dot If true, partial_dot is uint32_t*; if false, uint16_t*.
     * @param sum_q_f32 Pre-converted float array (caller converts uint32_t→float once per block).
     */
    template<bool U32Dot = false>
    static void RabitQCorrection(
        const void* partial_dot,
        float c1j, float c2j, float c34j, float qr_j,
        const float* sum_q_f32,
        const float* or_c_l2sqr,
        const float* dp_mult,
        float* out_partial_l2,
        size_t blk_count
    ) {
        for (size_t k = 0; k < blk_count; ++k) {
            float dot_f;
            if constexpr (U32Dot) {
                dot_f = static_cast<float>(static_cast<const uint32_t*>(partial_dot)[k]);
            } else {
                dot_f = static_cast<float>(static_cast<const uint16_t*>(partial_dot)[k]);
            }
            const float fdt = c1j * dot_f + c2j * sum_q_f32[k] - c34j;
            out_partial_l2[k] = or_c_l2sqr[k] + qr_j
                              - 2.0f * dp_mult[k] * fdt;
        }
    }

    /// kPerm0 interleaving used by nibble-split TransposeBlock.
    static constexpr int kPerm0[16] = {
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
    };

    template<bool WideAdd = false>
    static void ScanBlock(
        const uint8_t* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* out_dot,
        size_t blk_count
    ) {
        (void)sizeof(WideAdd);
        (void)blk_count;
        std::memset(out_dot, 0, kBlockSize * sizeof(uint16_t));
        for (size_t b = 0; b < binary_bytes; ++b) {
            const uint8_t* lut_lo = lut + b * 32;
            const uint8_t* lut_hi = lut + b * 32 + 16;
            const uint8_t* row = packed + b * kBlockSize;
            for (int j = 0; j < 16; ++j) {
                const uint8_t lo_byte = row[j];
                const uint8_t hi_byte = row[j + 16];
                const int vA = kPerm0[j];
                const int vB = kPerm0[j] + 16;
                out_dot[vA] += static_cast<uint16_t>(lut_lo[lo_byte & 0x0F])
                             + static_cast<uint16_t>(lut_hi[hi_byte & 0x0F]);
                out_dot[vB] += static_cast<uint16_t>(lut_lo[lo_byte >> 4])
                             + static_cast<uint16_t>(lut_hi[hi_byte >> 4]);
            }
        }
    }
};

} // namespace skmeans
