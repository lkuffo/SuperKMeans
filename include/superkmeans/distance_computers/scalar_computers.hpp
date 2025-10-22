#ifndef SUPERKMEANS_SCALAR_COMPUTERS_HPP
#define SUPERKMEANS_SCALAR_COMPUTERS_HPP

#include <iostream>

#include "superkmeans/common.h"

namespace skmeans {

template <DistanceFunction alpha, Quantization q> class ScalarComputer {};

template <> class ScalarComputer<l2, u8> {};

template <> class ScalarComputer<l2, f32> {
  public:
    using distance_t = skmeans_distance_t<f32>;
    using data_t     = skmeans_value_t<f32>;

    // Defer to the scalar kernel
    template <bool SKIP_PRUNED>
    static void VerticalPruning(const data_t* SKM_RESTRICT query, const data_t* SKM_RESTRICT data,
                                size_t n_vectors, size_t total_vectors, size_t start_dimension,
                                size_t end_dimension, distance_t* distances_p,
                                const uint32_t* pruning_positions = nullptr) {
        size_t dimensions_jump_factor = total_vectors;
        for (size_t dimension_idx = start_dimension; dimension_idx < end_dimension;
             ++dimension_idx) {
            uint32_t true_dimension_idx        = dimension_idx;
            size_t   offset_to_dimension_start = true_dimension_idx * dimensions_jump_factor;
            for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
                auto true_vector_idx = vector_idx;
                if constexpr (SKIP_PRUNED) {
                    true_vector_idx = pruning_positions[vector_idx];
                }
                distance_t to_multiply =
                    query[true_dimension_idx] - data[offset_to_dimension_start + true_vector_idx];
                distances_p[true_vector_idx] += to_multiply * to_multiply;
            }
        }
    }

    // Defer to the scalar kernel
    static void Vertical(const data_t* SKM_RESTRICT query, const data_t* SKM_RESTRICT data,
                         size_t start_dimension, size_t end_dimension, distance_t* distances_p) {
        for (size_t dim_idx = start_dimension; dim_idx < end_dimension; dim_idx++) {
            size_t dimension_idx             = dim_idx;
            size_t offset_to_dimension_start = dimension_idx * VECTOR_CHUNK_SIZE;
            for (size_t vector_idx = 0; vector_idx < VECTOR_CHUNK_SIZE; ++vector_idx) {
                distance_t to_multiply =
                    query[dimension_idx] - data[offset_to_dimension_start + vector_idx];
                distances_p[vector_idx] += to_multiply * to_multiply;
                //                if constexpr (L_ALPHA == IP){
                //                    distances_p[vector_idx] -= 2 * query[dimension_idx] *
                //                    data[offset_to_dimension_start + vector_idx];
                //                }
            }
        }
    }

    static distance_t Horizontal(const data_t* SKM_RESTRICT vector1,
                                 const data_t* SKM_RESTRICT vector2, size_t num_dimensions) {
        distance_t distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance_t to_multiply = vector1[dimension_idx] - vector2[dimension_idx];
            distance += to_multiply * to_multiply;
        }
        return distance;
    };
};

template <> class ScalarComputer<dp, f32> {
  public:
    using distance_t = skmeans_distance_t<f32>;
    using data_t     = skmeans_value_t<f32>;

    // Defer to the scalar kernel
    template <bool SKIP_PRUNED>
    static void VerticalPruning(const data_t* SKM_RESTRICT query, const data_t* SKM_RESTRICT data,
                                size_t n_vectors, size_t total_vectors, size_t start_dimension,
                                size_t end_dimension, distance_t* distances_p,
                                const uint32_t* pruning_positions = nullptr) {
        // TODO
    }

    // Defer to the scalar kernel
    static void Vertical(const data_t* SKM_RESTRICT query, const data_t* SKM_RESTRICT data,
                         size_t start_dimension, size_t end_dimension, distance_t* distances_p) {
        // TODO
    }

    static distance_t Horizontal(const data_t* SKM_RESTRICT vector1,
                                 const data_t* SKM_RESTRICT vector2, size_t num_dimensions) {
        distance_t distance = 0.0;
        for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
            distance += vector1[dimension_idx] * vector2[dimension_idx];
        }
        return distance;
    };
};

} // namespace skmeans

#endif // SUPERKMEANS_SCALAR_COMPUTERS_HPP
