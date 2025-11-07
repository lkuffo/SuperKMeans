#ifndef SUPERKMEANS_AVX512_COMPUTERS_HPP
#define SUPERKMEANS_AVX512_COMPUTERS_HPP

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/scalar_computers.hpp"

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class SIMDComputer {};

template <>
class SIMDComputer<l2, u8> {
  public:
    using distance_t = skmeans_distance_t<u8>;
    using data_t = skmeans_value_t<u8>;

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
class SIMDComputer<l2, f32> {
  public:
    using distance_t = skmeans_distance_t<f32>;
    using data_t = DataType_t<f32>;
    using scalar_computer = ScalarComputer<l2, f32>;

    alignas(64) static distance_t pruning_distances_tmp[4096];

    static void
    GatherDistances(size_t n_vectors, distance_t* distances_p, const uint32_t* pruning_positions) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            auto true_vector_idx = pruning_positions[vector_idx];
            pruning_distances_tmp[vector_idx] = distances_p[true_vector_idx];
        }
    }

    static void GatherBasedKernel(
        const data_t* SKM_RESTRICT query,
        const data_t* SKM_RESTRICT data,
        size_t n_vectors,
        size_t total_vectors,
        size_t start_dimension,
        size_t end_dimension,
        distance_t* distances_p,
        const uint32_t* pruning_positions = nullptr
    ) {
        GatherDistances(n_vectors, distances_p, pruning_positions);
        __m512 data_vec, d_vec, cur_dist_vec;
        __m256 data_vec_m256, d_vec_m256, cur_dist_vec_m256;
        // Then we move data to be sequential
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension;
             ++dimension_idx) {
            uint32_t true_dimension_idx = dimension_idx;
            __m512 query_vec;
            query_vec = _mm512_set1_ps(query[true_dimension_idx]);
            //            if constexpr (L_ALPHA == IP){
            //                query_vec = _mm512_set1_ps(-2 * query[true_dimension_idx]);
            //            }
            size_t offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            const float* tmp_data = data + offset_to_dimension_start;
            // Now we do the sequential distance calculation loop which would use SIMD
            // Up to 16
            size_t i = 0;
            for (; i + 16 < n_vectors; i += 16) {
                cur_dist_vec = _mm512_load_ps(&pruning_distances_tmp[i]);
                data_vec = _mm512_i32gather_ps(
                    _mm512_load_epi32(&pruning_positions[i]), tmp_data, sizeof(distance_t)
                );
                d_vec = _mm512_sub_ps(data_vec, query_vec);
                cur_dist_vec = _mm512_fmadd_ps(d_vec, d_vec, cur_dist_vec);
                _mm512_store_ps(&pruning_distances_tmp[i], cur_dist_vec);
            }
            __m256 query_vec_m256;
            query_vec_m256 = _mm256_set1_ps(query[true_dimension_idx]);
            //            if constexpr (L_ALPHA == IP){
            //                query_vec_m256 = _mm256_set1_ps(-2 * query[true_dimension_idx]);
            //            }
            // Up to 8
            for (; i + 8 < n_vectors; i += 8) {
                cur_dist_vec_m256 = _mm256_load_ps(&pruning_distances_tmp[i]);
                data_vec_m256 = _mm256_i32gather_ps(
                    tmp_data, _mm256_load_epi32(&pruning_positions[i]), sizeof(distance_t)
                );
                d_vec_m256 = _mm256_sub_ps(data_vec_m256, query_vec_m256);
                cur_dist_vec_m256 = _mm256_fmadd_ps(d_vec_m256, d_vec_m256, cur_dist_vec_m256);
                _mm256_store_ps(&pruning_distances_tmp[i], cur_dist_vec_m256);
            }
            // Tail
            for (; i < n_vectors; i++) {
                float to_multiply = query[true_dimension_idx] - tmp_data[pruning_positions[i]];
                pruning_distances_tmp[i] += to_multiply * to_multiply;
            }
        }
        // We now move distances back
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            auto true_vector_idx = pruning_positions[vector_idx];
            distances_p[true_vector_idx] = pruning_distances_tmp[vector_idx];
        }
    }

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
        // SIMD is less efficient when looping on the array of not-yet pruned vectors
        // A way to improve the performance by ~20% is using a GATHER intrinsic. However this only
        // works on Intel microarchs. In AMD (Zen 4, Zen 3) using a GATHER is shooting ourselves in
        // the foot (~80 uops)
        // __AVX512FP16__ macro let us detect Intel architectures (from Sapphire Rapids onwards)
#if false && defined(__AVX512FP16__)
        if (n_vectors >= 8) {
            GatherBasedKernel(
                    query, data, n_vectors, total_vectors, start_dimension, end_dimension,
                    distances_p
            );
            return;
        }
#endif
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

} // namespace skmeans

#endif // SUPERKMEANS_AVX512_COMPUTERS_HPP
