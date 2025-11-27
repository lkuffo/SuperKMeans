#pragma once

#include <algorithm>
#include <cassert>
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/pdx_ivf.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/utils.h"

namespace skmeans {

/******************************************************************
 * PDXearch
 * Implements our algorithm for vertical pruning
 ******************************************************************/
template <
    Quantization q = Quantization::f32,
    class Index = IndexPDXIVF<q>,
    DistanceFunction alpha = DistanceFunction::l2>
class PDXearch {
  public:
    using DISTANCES_TYPE = skmeans_distance_t<q>;
    using DATA_TYPE = skmeans_value_t<q>;
    using INDEX_TYPE = Index;
    using CLUSTER_TYPE = Cluster<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;
    using Pruner = ADSamplingPruner<q>;

    Pruner& pruner;
    INDEX_TYPE& pdx_data;

    PDXearch(INDEX_TYPE& data_index, Pruner& pruner) : pruner(pruner), pdx_data(data_index) {}

  protected:
    const float selectivity_threshold = 0.80;

    template <Quantization Q = q>
    SKM_NO_INLINE void GetPruningThreshold(
        KNNCandidate<Q>& best_candidate,
        skmeans_distance_t<Q>& pruning_threshold,
        uint32_t current_dimension_idx
    ) {
        pruning_threshold = pruner.template GetPruningThreshold<Q>(best_candidate, current_dimension_idx);
    }

    template <Quantization Q = q>
    SKM_NO_INLINE void EvaluatePruningPredicateScalar(
        uint32_t& n_pruned,
        size_t n_vectors,
        skmeans_distance_t<Q>* pruning_distances,
        const skmeans_distance_t<Q> pruning_threshold
    ) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    }

    template <Quantization Q = q>
    SKM_NO_INLINE void EvaluatePruningPredicateOnPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q> pruning_threshold,
        skmeans_distance_t<Q>* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
            n_vectors_not_pruned +=
                pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
        }
    }

    template <Quantization Q = q>
    SKM_NO_INLINE void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q> pruning_threshold,
        skmeans_distance_t<Q>* pruning_distances
    ) {
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
    }

    // On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors
    // is low
    template <Quantization Q = q>
    void Warmup(
        const skmeans_value_t<Q>* SKM_RESTRICT query,
        const skmeans_value_t<Q>* SKM_RESTRICT data,
        const size_t n_vectors,
        float tuples_threshold,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q>* pruning_distances,
        skmeans_distance_t<Q>& pruning_threshold,
        KNNCandidate<Q>& best_candidate,
        uint32_t& current_dimension_idx
    ) {
        size_t cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit = std::ceil(1.0 * tuples_threshold * n_vectors);
        uint32_t n_tuples_to_prune = 0;
        GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
        EvaluatePruningPredicateScalar<Q>(
            n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold
        );
        while (1.0 * n_tuples_to_prune < tuples_needed_to_exit &&
               current_dimension_idx < pdx_data.num_vertical_dimensions) {
            size_t last_dimension_to_fetch = std::min(
                current_dimension_idx + DIMENSIONS_FETCHING_SIZES[cur_subgrouping_size_idx],
                pdx_data.num_vertical_dimensions
            );
            DistanceComputer<alpha, Q>::Vertical(
                query,
                data,
                n_vectors,
                n_vectors,
                current_dimension_idx,
                last_dimension_to_fetch,
                pruning_distances,
                pruning_positions
            );
            current_dimension_idx = last_dimension_to_fetch;
            cur_subgrouping_size_idx += 1;
            GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
            n_tuples_to_prune = 0;
            EvaluatePruningPredicateScalar<Q>(
                n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold
            );
        }
    }

    // We scan only the not-yet pruned vectors
    template <Quantization Q = q>
    SKM_NO_INLINE
    void Prune(
        const skmeans_value_t<Q>* SKM_RESTRICT query,
        const skmeans_value_t<Q>* SKM_RESTRICT data,
        const size_t n_vectors,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q>* pruning_distances,
        skmeans_distance_t<Q>& pruning_threshold,
        KNNCandidate<Q>& best_candidate,
        size_t& n_vectors_not_pruned,
        uint32_t& current_dimension_idx,
        const uint32_t* vector_indices,
        const uint32_t prev_top_1,
        const skmeans_value_t<Q>* SKM_RESTRICT aux_data = nullptr,
        size_t* initial_not_pruned_out = nullptr
    ) {
        GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
        InitPositionsArray<Q>(
            n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances
        );
        // Record the initial n_vectors_not_pruned if requested
        if (initial_not_pruned_out != nullptr) {
            *initial_not_pruned_out = n_vectors_not_pruned;
        }
        // Early exit if the only remaining point is the one that was initially the best candidate
        if (n_vectors_not_pruned == 1 && vector_indices[pruning_positions[0]] == prev_top_1) {
            n_vectors_not_pruned = 0;
            return;
        }
        size_t cur_n_vectors_not_pruned = 0;
        size_t current_vertical_dimension = current_dimension_idx;
        size_t current_horizontal_dimension = 0;
        while (pdx_data.num_horizontal_dimensions && n_vectors_not_pruned &&
               current_horizontal_dimension < pdx_data.num_horizontal_dimensions) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            size_t offset_data = (pdx_data.num_vertical_dimensions * n_vectors) +
                                 (current_horizontal_dimension * n_vectors);
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                __builtin_prefetch(data + data_pos, 0, 2);
            }
            size_t offset_query = pdx_data.num_vertical_dimensions + current_horizontal_dimension;
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
                    query + offset_query, data + data_pos, H_DIM_SIZE
                );
            }
            current_horizontal_dimension += H_DIM_SIZE;
            current_dimension_idx += H_DIM_SIZE;
            GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
            assert(
                current_dimension_idx == current_vertical_dimension +
                current_horizontal_dimension
            );
            EvaluatePruningPredicateOnPositionsArray<Q>(
                cur_n_vectors_not_pruned,
                n_vectors_not_pruned,
                pruning_positions,
                pruning_threshold,
                pruning_distances
            );
        }
        // GO THROUGH THE REST IN THE VERTICAL
        while (n_vectors_not_pruned && current_vertical_dimension < pdx_data.num_vertical_dimensions
        ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            if (aux_data == nullptr) {
                size_t last_dimension_to_test_idx = std::min(
                    current_vertical_dimension + H_DIM_SIZE,
                    (size_t) pdx_data.num_vertical_dimensions
                );
                DistanceComputer<alpha, Q>::VerticalPruning(
                    query,
                    data,
                    cur_n_vectors_not_pruned,
                    n_vectors,
                    current_vertical_dimension,
                    last_dimension_to_test_idx,
                    pruning_distances,
                    pruning_positions
                );
                current_dimension_idx =
                    std::min(current_dimension_idx + H_DIM_SIZE, (size_t) pdx_data.num_dimensions);
                current_vertical_dimension = std::min(
                    (uint32_t) (current_vertical_dimension + H_DIM_SIZE),
                    pdx_data.num_vertical_dimensions
                );
            } else { // !We have the data also in the Horizontal layout
                // We go till the end
                size_t dimensions_left =
                    pdx_data.num_vertical_dimensions - current_vertical_dimension;
                // std::cout << "Dims left" << dimensions_left << std::endl;
                size_t offset_query = current_vertical_dimension;
                for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                    size_t v_idx = pruning_positions[vector_idx];
                    auto data_pos = aux_data + (v_idx * pdx_data.num_vertical_dimensions) +
                                    current_vertical_dimension;
                    __builtin_prefetch(data_pos, 0, 1);
                }
                for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                    size_t v_idx = pruning_positions[vector_idx];
                    auto data_pos = aux_data + (v_idx * pdx_data.num_vertical_dimensions) +
                                    current_vertical_dimension;
                    pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
                        query + offset_query, data_pos, dimensions_left
                    );
                }
                current_dimension_idx = pdx_data.num_dimensions;
                current_vertical_dimension = pdx_data.num_vertical_dimensions;
            }
            assert(
                current_dimension_idx == current_vertical_dimension + current_horizontal_dimension
            );
            GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
            EvaluatePruningPredicateOnPositionsArray<Q>(
                cur_n_vectors_not_pruned,
                n_vectors_not_pruned,
                pruning_positions,
                pruning_threshold,
                pruning_distances
            );
            if (current_dimension_idx == pdx_data.num_dimensions)
                break;
        }
    }

    template <Quantization Q = q>
    void SetBestCandidate(
        const uint32_t* vector_indices,
        size_t n_vectors,
        const uint32_t* pruning_positions,
        const skmeans_distance_t<Q>* pruning_distances,
        KNNCandidate<Q>& best_candidate
    ) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = pruning_positions[position_idx];
            skmeans_distance_t<Q> current_distance = pruning_distances[index];
            if (current_distance < best_candidate.distance) {
                best_candidate.distance = current_distance;
                best_candidate.index = vector_indices[index];
            }
        }
    }

    void BuildResultSet(KNNCandidate_t& best_candidate) {
        // We return distances in the original domain
        if constexpr (q == Quantization::u8) {
            float inverse_scale_factor = 1.0f / pdx_data.scale_factor;
            inverse_scale_factor = inverse_scale_factor * inverse_scale_factor;
            best_candidate.distance = best_candidate.distance * inverse_scale_factor;
        }
    }

  public:
    /******************************************************************
     * Search methods
     ******************************************************************/
    KNNCandidate_t Top1SearchWithThreshold(
        const float* SKM_RESTRICT query,
        const float prev_pruning_threshold,
        const uint32_t prev_top_1,
        uint32_t& pruned_at_accum,
        const size_t start_cluster,
        const size_t end_cluster
    ) {
        alignas(64) thread_local DISTANCES_TYPE pruning_distances[PDX_VECTOR_SIZE];
        alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];
        DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        size_t n_vectors_not_pruned = 0;
        uint32_t current_dimension_idx = 0;
        size_t current_cluster = 0;

        // Setup previous top1
        pruning_threshold = prev_pruning_threshold;
        auto top_embedding = KNNCandidate<q>{};
        top_embedding.index = prev_top_1;
        top_embedding.distance = prev_pruning_threshold;
        // PDXearch core
        current_dimension_idx = 0;
        for (size_t cluster_idx = start_cluster; cluster_idx < end_cluster; ++cluster_idx) {
            current_cluster = cluster_idx;
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            Warmup(
                query,
                cluster.data,
                cluster.num_embeddings,
                selectivity_threshold,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                top_embedding,
                current_dimension_idx
            );
            pruned_at_accum += current_dimension_idx;
            Prune(
                query,
                cluster.data,
                cluster.num_embeddings,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                top_embedding,
                n_vectors_not_pruned,
                current_dimension_idx,
                cluster.indices,
                prev_top_1,
                cluster.aux_hor_data
            );
            if (n_vectors_not_pruned) {
                SetBestCandidate(
                    cluster.indices,
                    n_vectors_not_pruned,
                    pruning_positions,
                    pruning_distances,
                    top_embedding
                );
            }
        }
        BuildResultSet(top_embedding);
        return top_embedding;
    }

    SKM_NO_INLINE
    KNNCandidate_t Top1PartialSearchWithThresholdAndPartialDistances(
        const float* SKM_RESTRICT query,
        const float prev_pruning_threshold,
        const uint32_t prev_top_1,
        DISTANCES_TYPE* partial_pruning_distances,
        const uint32_t computed_distance_until,
        const size_t start_cluster,
        const size_t end_cluster,
        size_t* initial_not_pruned_accum = nullptr
    ) {
        alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];
        DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        size_t n_vectors_not_pruned = 0;
        uint32_t current_dimension_idx = computed_distance_until;
        size_t current_cluster = 0;

        // Setup previous top1
        pruning_threshold = prev_pruning_threshold;
        auto top_embedding = KNNCandidate<q>{};
        top_embedding.index = prev_top_1;
        top_embedding.distance = prev_pruning_threshold;
        // PDXearch core
        size_t data_offset = 0;
        for (size_t cluster_idx = start_cluster; cluster_idx < end_cluster; ++cluster_idx) {
            current_dimension_idx = computed_distance_until;

            auto pruning_distances = partial_pruning_distances + data_offset;
            current_cluster = cluster_idx;
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            data_offset += cluster.num_embeddings;
            size_t initial_not_pruned = 0;
            Prune(
                query,
                cluster.data,
                cluster.num_embeddings,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                top_embedding,
                n_vectors_not_pruned,
                current_dimension_idx,
                cluster.indices,
                prev_top_1,
                cluster.aux_hor_data,
                &initial_not_pruned
            );
            // Accumulate the initial not-pruned count for this cluster
            if (initial_not_pruned_accum != nullptr) {
                *initial_not_pruned_accum += initial_not_pruned;
            }
            if (n_vectors_not_pruned) {
                SetBestCandidate(
                    cluster.indices,
                    n_vectors_not_pruned,
                    pruning_positions,
                    pruning_distances,
                    top_embedding
                );
            }
        }
        BuildResultSet(top_embedding);
        return top_embedding;
    }

};

} // namespace skmeans
