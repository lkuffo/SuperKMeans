#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/scalar_computers.h"

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::u8> {
  public:
    using distance_t = skmeans_distance_t<Quantization::u8>;
    using data_t = skmeans_value_t<Quantization::u8>;

    template <bool SKIP_PRUNED>
    static void VerticalPruning(
        const data_t* SKM_RESTRICT query,
        const data_t* SKM_RESTRICT data,
        size_t n_vectors,
        size_t total_vectors,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p,
        const uint32_t* pruning_positions = nullptr
    ) {
        __m512i res;
        __m512i vec2_u8;
        __m512i vec1_u8;
        __m512i diff_u8;
        __m256i y_res;
        __m256i y_vec2_u8;
        __m256i y_vec1_u8;
        __m256i y_diff_u8;
        uint32_t* query_grouped = (uint32_t*) query;
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx += 4) {
            uint32_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * total_vectors;
            size_t i = 0;
            if constexpr (!SKIP_PRUNED) {
                // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1
                // word)
                uint32_t query_value = query_grouped[dimension_idx / 4];
                // And then broadcast it to the register
                vec1_u8 = _mm512_set1_epi32(query_value);
                for (; i + 16 <= n_vectors; i += 16) {
                    // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
                    res = _mm512_load_si512(&distances_p[i]);
                    vec2_u8 = _mm512_loadu_si512(&data[offset_to_dimension_start + i * 4]
                    ); // This 4 is because everytime I read 4 dimensions
                    diff_u8 = _mm512_or_si512(
                        _mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8)
                    );
                    _mm512_store_epi32(
                        &distances_p[i], _mm512_dpbusds_epi32(res, diff_u8, diff_u8)
                    );
                }
                y_vec1_u8 = _mm256_set1_epi32(query_value);
                for (; i + 8 <= n_vectors; i += 8) {
                    // Read 32 bytes of data (32 values) with 4 dimensions of 8 vectors
                    y_res = _mm256_load_epi32(&distances_p[i]);
                    y_vec2_u8 = _mm256_loadu_epi8(&data[offset_to_dimension_start + i * 4]
                    ); // This 4 is because everytime I read 4 dimensions
                    y_diff_u8 = _mm256_or_si256(
                        _mm256_subs_epu8(y_vec1_u8, y_vec2_u8),
                        _mm256_subs_epu8(y_vec2_u8, y_vec1_u8)
                    );
                    _mm256_store_epi32(
                        &distances_p[i], _mm256_dpbusds_epi32(y_res, y_diff_u8, y_diff_u8)
                    );
                }
            }
            // rest
            for (; i < n_vectors; ++i) {
                size_t vector_idx = i;
                if constexpr (SKIP_PRUNED) {
                    vector_idx = pruning_positions[vector_idx];
                }
                int to_multiply_a =
                    query[dimension_idx] - data[offset_to_dimension_start + (vector_idx * 4)];
                int to_multiply_b = query[dimension_idx + 1] -
                                    data[offset_to_dimension_start + (vector_idx * 4) + 1];
                int to_multiply_c = query[dimension_idx + 2] -
                                    data[offset_to_dimension_start + (vector_idx * 4) + 2];
                int to_multiply_d = query[dimension_idx + 3] -
                                    data[offset_to_dimension_start + (vector_idx * 4) + 3];
                distances_p[vector_idx] +=
                    (to_multiply_a * to_multiply_a) + (to_multiply_b * to_multiply_b) +
                    (to_multiply_c * to_multiply_c) + (to_multiply_d * to_multiply_d);
            }
        }
    }

    static void Vertical(
        const data_t* SKM_RESTRICT query,
        const data_t* SKM_RESTRICT data,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p
    ) {
        __m512i res[4];
        const uint32_t* query_grouped = (uint32_t*) query;
        // Load 64 initial values
        for (size_t i = 0; i < 4; ++i) {
            res[i] = _mm512_load_si512(&distances_p[i * 16]);
        }
        // Compute l2
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx += 4) {
            const uint32_t dimension_idx = dim_idx;
            // To load the query efficiently I will load it as uint32_t (4 bytes packed in 1 word)
            const uint32_t query_value = query_grouped[dimension_idx / 4];
            // And then broadcast it to the register
            const __m512i vec1_u8 = _mm512_set1_epi32(query_value);
            const size_t offset_to_dimension_start = dimension_idx * VECTOR_CHUNK_SIZE;
            for (int i = 0; i < 4; ++i) { // total: 64 vectors (4 iterations of 16 vectors) * 4
                                          // dimensions each (at 1 byte per value = 2048-bits)
                // Read 64 bytes of data (64 values) with 4 dimensions of 16 vectors
                const __m512i vec2_u8 =
                    _mm512_loadu_si512(&data[offset_to_dimension_start + i * 64]);
                const __m512i diff_u8 = _mm512_or_si512(
                    _mm512_subs_epu8(vec1_u8, vec2_u8), _mm512_subs_epu8(vec2_u8, vec1_u8)
                );
                // I can use this asymmetric dot product as my values are actually 7-bit
                // Hence, the [sign] properties of the second operand is ignored
                // As results will never be negative, it can be stored on res[i] without issues
                // and it saturates to MAX_INT
                res[i] = _mm512_dpbusds_epi32(res[i], diff_u8, diff_u8);
            }
        }
        // Store results back
        for (int i = 0; i < 4; ++i) {
            _mm512_store_epi32(&distances_p[i * 16], res[i]);
        }
    }

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m512i d2_i32_vec = _mm512_setzero_si512();
        __m512i a_u8_vec, b_u8_vec;

    simsimd_l2sq_u8_ice_cycle:
        if (num_dimensions < 64) {
            const __mmask64 mask = (__mmask64) _bzhi_u64(0xFFFFFFFFFFFFFFFF, num_dimensions);
            a_u8_vec = _mm512_maskz_loadu_epi8(mask, vector1);
            b_u8_vec = _mm512_maskz_loadu_epi8(mask, vector2);
            num_dimensions = 0;
        } else {
            a_u8_vec = _mm512_loadu_si512(vector1);
            b_u8_vec = _mm512_loadu_si512(vector2);
            vector1 += 64, vector2 += 64, num_dimensions -= 64;
        }

        // Substracting unsigned vectors in AVX-512 is done by saturating subtraction:
        __m512i d_u8_vec = _mm512_or_si512(
            _mm512_subs_epu8(a_u8_vec, b_u8_vec), _mm512_subs_epu8(b_u8_vec, a_u8_vec)
        );

        // Multiply and accumulate at `int8` level which are actually uint7, accumulate at `int32`
        // level:
        d2_i32_vec = _mm512_dpbusds_epi32(d2_i32_vec, d_u8_vec, d_u8_vec);
        if (num_dimensions)
            goto simsimd_l2sq_u8_ice_cycle;
        return _mm512_reduce_add_epi32(d2_i32_vec);
    };
};

template <>
class SIMDComputer<skmeans::DistanceFunction::l2, Quantization::f32> {
  public:
    using distance_t = skmeans_distance_t<Quantization::f32>;
    using data_t = skmeans_value_t<Quantization::f32>;
    using scalar_computer = ScalarComputer<skmeans::DistanceFunction::l2, Quantization::f32>;

    // Defer to the scalar kernel
    template <bool SKIP_PRUNED>
    static void VerticalPruning(
        const data_t* SKM_RESTRICT query,
        const data_t* SKM_RESTRICT data,
        size_t n_vectors,
        size_t total_vectors,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p,
        const uint32_t* pruning_positions = nullptr
    ) {
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension;
             ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = vector_idx;
                if constexpr (SKIP_PRUNED) {
                    true_vector_idx = pruning_positions[vector_idx];
                }
                float to_multiply =
                    query[true_dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
                distances_p[true_vector_idx] += to_multiply * to_multiply;
            }
        }
    }

    // Defer to the scalar kernel
    static void Vertical(
        const data_t* SKM_RESTRICT query,
        const data_t* SKM_RESTRICT data,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p
    ) {
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * VECTOR_CHUNK_SIZE;
            for (size_t vector_idx = 0; vector_idx < VECTOR_CHUNK_SIZE; ++vector_idx) {
                float to_multiply =
                    query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                distances_p[vector_idx] += to_multiply * to_multiply;
            }
        }
    }

    static distance_t Horizontal(
        const data_t* SKM_RESTRICT vector1,
        const data_t* SKM_RESTRICT vector2,
        size_t num_dimensions
    ) {
        __m512 d2_vec = _mm512_setzero();
        __m512 a_vec, b_vec;
    simsimd_l2sq_f32_skylake_cycle:
        if (num_dimensions < 16) {
            __mmask16 mask = (__mmask16) _bzhi_u32(0xFFFFFFFF, num_dimensions);
            a_vec = _mm512_maskz_loadu_ps(mask, vector1);
            b_vec = _mm512_maskz_loadu_ps(mask, vector2);
            num_dimensions = 0;
        } else {
            a_vec = _mm512_loadu_ps(vector1);
            b_vec = _mm512_loadu_ps(vector2);
            vector1 += 16, vector2 += 16, num_dimensions -= 16;
        }
        __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
        d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
        if (num_dimensions)
            goto simsimd_l2sq_f32_skylake_cycle;

        // _simsimd_reduce_f32x16_skylake
        __m512 x =
            _mm512_add_ps(d2_vec, _mm512_shuffle_f32x4(d2_vec, d2_vec, _MM_SHUFFLE(0, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(
            _mm512_add_ps(x, _mm512_shuffle_f32x4(x, x, _MM_SHUFFLE(0, 0, 0, 1)))
        );
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
    };
};


template <Quantization q>
class SIMDUtilsComputer {};

template <>
class SIMDUtilsComputer<Quantization::f32> {
  public:
    using data_t = skmeans_value_t<Quantization::f32>;

    /**
     * @brief Flip sign of floats based on a mask using AVX-512 (single vector).
     * @param data Input vector (d elements)
     * @param out Output vector (can be same as data for in-place)
     * @param masks Bitmask array (0x80000000 to flip, 0 to keep)
     * @param d Number of dimensions
     */
    static void FlipSign(
        const data_t* data,
        data_t* out,
        const uint32_t* masks,
        size_t d
    ) {
        size_t j = 0;
        // AVX-512: process 16 floats at a time
        for (; j + 16 <= d; j += 16) {
            __m512 vec = _mm512_loadu_ps(data + j);
            __m512i mask = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(masks + j));
            __m512i vec_i = _mm512_castps_si512(vec);
            vec_i = _mm512_xor_si512(vec_i, mask);
            _mm512_storeu_ps(out + j, _mm512_castsi512_ps(vec_i));
        }
        // AVX2 tail: process 8 floats
        for (; j + 8 <= d; j += 8) {
            __m256 vec = _mm256_loadu_ps(data + j);
            __m256i mask_avx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(masks + j));
            __m256i vec_i = _mm256_castps_si256(vec);
            vec_i = _mm256_xor_si256(vec_i, mask_avx);
            _mm256_storeu_ps(out + j, _mm256_castsi256_ps(vec_i));
        }
        // Scalar tail
        auto data_bits = reinterpret_cast<const uint32_t*>(data);
        auto out_bits = reinterpret_cast<uint32_t*>(out);
        for (; j < d; ++j) {
            out_bits[j] = data_bits[j] ^ masks[j];
        }
    }

    /**
     * @brief Initializes positions array with indices of non-pruned vectors using AVX-512.
     *
     * Optimized for cases where only ~2% of vectors pass the threshold test.
     * Processes 16 floats at a time and uses vpcompressd for efficient scatter.
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
        size_t vector_idx = 0;

        constexpr size_t k_simd_width = 16;
        const size_t n_vectors_simd = (n_vectors / k_simd_width) * k_simd_width;

        __m512 threshold_vec = _mm512_set1_ps(pruning_threshold);

        // Process 16 elements at a time
        for (; vector_idx < n_vectors_simd; vector_idx += k_simd_width) {
            __m512 distances = _mm512_loadu_ps(pruning_distances + vector_idx);
            __mmask16 cmp_mask = _mm512_cmp_ps_mask(distances, threshold_vec, _CMP_LT_OQ);

            // Branch hint: likely that no elements passed (98% of the time)
            if (SKM_UNLIKELY(cmp_mask)) {
                // At least one element passed - use vpcompressd to store compacted indices
                // Create vector of indices [vector_idx, vector_idx+1, ..., vector_idx+15]
                __m512i indices = _mm512_add_epi32(
                    _mm512_set1_epi32(vector_idx),
                    _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
                );

                // Compress and store indices where mask is true
                _mm512_mask_compressstoreu_epi32(
                    pruning_positions + n_vectors_not_pruned,
                    cmp_mask,
                    indices
                );

                n_vectors_not_pruned += _mm_popcnt_u32(cmp_mask);
            }
        }
        // Tail
        for (; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }
};

} // namespace skmeans
