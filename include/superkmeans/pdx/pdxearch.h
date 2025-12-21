#pragma once

#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/pdx_ivf.h"
#include "superkmeans/pdx/utils.h"
#include <algorithm>
#include <cassert>

namespace skmeans {

/**
 * @brief PDXearch - Efficient Top-1 nearest neighbor search with early termination.
 *
 * Implements the PDXearch algorithm for finding the nearest neighbor using the PDX
 * data layout combined with ADSampling-based pruning. In this lightweight implementation,
 * we skip the warmup phase and directly use the pre-computed partial distances.
 *
 * @tparam q Quantization type (f32 or u8)
 * @tparam Index Index type (typically IndexPDXIVF<q>)
 * @tparam alpha Distance function (l2 or dp)
 */
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

    Pruner& pruner;       ///< Reference to the ADSampling pruner
    INDEX_TYPE& pdx_data; ///< Reference to the PDX index data

    /**
     * @brief Constructs a PDXearch instance.
     *
     * @param data_index Reference to the PDX index containing the data
     * @param pruner Reference to the ADSampling pruner for threshold computation
     */
    PDXearch(INDEX_TYPE& data_index, Pruner& pruner) : pruner(pruner), pdx_data(data_index) {}

  protected:
    const float selectivity_threshold =
        0.80; ///< Target fraction of vectors to prune in warmup phase

    /**
     * @brief Retrieves the pruning threshold from the pruner.
     */
    template <Quantization Q = q>
    SKM_NO_INLINE void GetPruningThreshold(
        KNNCandidate<Q>& best_candidate,
        skmeans_distance_t<Q>& pruning_threshold,
        uint32_t current_dimension_idx
    ) {
        pruning_threshold =
            pruner.template GetPruningThreshold<Q>(best_candidate, current_dimension_idx);
    }

    /**
     * @brief Updates the positions array to keep only non-pruned candidates.
     */
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

    /**
     * @brief Initializes the positions array with indices of non-pruned vectors.
     */
    template <Quantization Q = q>
    SKM_NO_INLINE void InitPositionsArray(
        size_t n_vectors,
        size_t& n_vectors_not_pruned,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q> pruning_threshold,
        skmeans_distance_t<Q>* pruning_distances
    ) {
        UtilsComputer<Q>::InitPositionsArray(
            n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances
        );
    }

    /**
     * @brief Prune phase: iteratively eliminates candidates using remaining dimensions.
     *
     * Processes horizontal dimensions first (more SIMD-friendly), then remaining vertical
     * dimensions. After each block, re-evaluates which candidates can be pruned.
     *
     * @tparam Q Quantization type
     * @param query Query vector
     * @param data PDX-formatted cluster data
     * @param n_vectors Number of vectors in the cluster
     * @param pruning_positions Array of candidate positions (compacted in place)
     * @param pruning_distances Array of partial distances (updated in place)
     * @param pruning_threshold Current pruning threshold (updated)
     * @param best_candidate Current best candidate
     * @param n_vectors_not_pruned Number of remaining candidates (updated)
     * @param current_dimension_idx Number of dimensions processed (updated)
     * @param vector_indices Mapping from local to global vector indices
     * @param prev_top_1 Previous best candidate index (for early exit)
     * @param aux_vertical_dimensions_in_horizontal_layout Optional auxiliary horizontal data for
     * vertical dimensions
     * @param initial_not_pruned_out Optional output for initial non-pruned count
     */
    template <Quantization Q = q>
    SKM_NO_INLINE void Prune(
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
        const skmeans_value_t<Q>* SKM_RESTRICT aux_vertical_dimensions_in_horizontal_layout =
            nullptr,
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
                current_dimension_idx == current_vertical_dimension + current_horizontal_dimension
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
            // !We have the data also in the Horizontal layout, so we go till the end
            size_t dimensions_left = pdx_data.num_vertical_dimensions - current_vertical_dimension;
            size_t offset_query = current_vertical_dimension;
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                auto data_pos = aux_vertical_dimensions_in_horizontal_layout +
                                (v_idx * pdx_data.num_vertical_dimensions) +
                                current_vertical_dimension;
                __builtin_prefetch(data_pos, 0, 1);
            }
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                auto data_pos = aux_vertical_dimensions_in_horizontal_layout +
                                (v_idx * pdx_data.num_vertical_dimensions) +
                                current_vertical_dimension;
                pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
                    query + offset_query, data_pos, dimensions_left
                );
            }
            current_dimension_idx = pdx_data.num_dimensions;
            current_vertical_dimension = pdx_data.num_vertical_dimensions;
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
    SKM_NO_INLINE void GPUPrune(
        const skmeans_value_t<Q>* SKM_RESTRICT query,
        const skmeans_value_t<Q>* SKM_RESTRICT data,
        const size_t n_vectors,
        uint32_t* pruning_positions, // Preallocated buffer[PDX_VECTOR_SIZE], undefined content
        skmeans_distance_t<Q>* pruning_distances, // all_distances_buf
        skmeans_distance_t<Q>& pruning_threshold, // initially dist_to_prev centroid, then this function updates it
        KNNCandidate<Q>& best_candidate, // initially prev centroid, then updated
        size_t& n_vectors_not_pruned, // initially 0, then updated 
        uint32_t& current_dimension_idx, //??
        const uint32_t* vector_indices,
        const uint32_t prev_top_1,
        const skmeans_value_t<Q>* SKM_RESTRICT aux_vertical_dimensions_in_horizontal_layout =
            nullptr,
        size_t* initial_not_pruned_out = nullptr
    ) {
			using distance_t = skmeans_distance_t<Quantization::f32>;
        //GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
				pruning_threshold = best_candidate.distance * pruner.ratios[current_dimension_idx];
				// ratios[num_dimensions] <- ratio accounts for the expected contribution of remaining dimensions to the total distance.
				// return best_candidate.distance * ratios[current_dimension_idx];
				// Basically, for the current dimension, speculatively decrease treshold (make it more tough)

        // InitPositionsArray<Q>(
        //     n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances
        // );
        n_vectors_not_pruned = 0;
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
        }
				// Intuition: Get all vectors who beat the threshold, put them in the first n_vectors_not_pruned positions 
				// of the pruning_positions buffer of size n_vectors
				//
        // n_vectors_not_pruned = 0;
				// SH: For each vector
        // for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
				// 		// assign last vector who beat the threshold to pruning_position
        //     pruning_positions[n_vectors_not_pruned] = vector_idx;
				//
				//     bool partial_distance_beats_treshold = pruning_distances[vector_idx] < pruning_threshold;
				//     if (partial_distance_beats_treshold) {
				//     	++n_vectors_not_pruned;
				//     }
        // }

        // Record the initial n_vectors_not_pruned if requested
        if (initial_not_pruned_out != nullptr) {
            *initial_not_pruned_out = n_vectors_not_pruned;
        }
        // Early exit if the only remaining point is the one that was initially the best candidate
        if (n_vectors_not_pruned == 1 && vector_indices[pruning_positions[0]] == prev_top_1) {
            n_vectors_not_pruned = 0;
            return;
        }

				// Kernel parameters
				// pdx_data.num_horizontal_dimensions
				// pdx_data.num_vertical_dimensions
				// n_vectors_not_pruned
				// n_vectors
				//
				// pruning_positions
				// pruning_distances
        size_t cur_n_vectors_not_pruned = 0;
        size_t current_vertical_dimension = current_dimension_idx;
				const bool has_horizontal_dimensions = pdx_data.num_horizontal_dimensions > 0;
				// SH Do not launch kernel if there is nothing to do
				if (has_horizontal_dimensions && n_vectors_not_pruned != 0) {
					// SH for every horizontal dimension until all vectors pruned
					for (size_t current_horizontal_dimension = 0; current_horizontal_dimension < pdx_data.num_horizontal_dimensions; current_horizontal_dimension += H_DIM_SIZE) {
							const bool pruned_all_vectors = n_vectors_not_pruned == 0;
							if (pruned_all_vectors) {
								break;
							}
							cur_n_vectors_not_pruned = n_vectors_not_pruned;
							size_t offset_data = (pdx_data.num_vertical_dimensions * n_vectors) +
																	 (current_horizontal_dimension * n_vectors);
							// SH: Calculate distance of query to all not pruned vectors in block for H_DIM_SIZE dimensions
							size_t offset_query = pdx_data.num_vertical_dimensions + current_horizontal_dimension;
							for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
									size_t v_idx = pruning_positions[vector_idx];
									size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
									// Increase partial_distance with rest of the horizontal distance

									distance_t adistance = 0.0;
									auto vector1 = query + offset_query;
									auto vector2 = data + data_pos;
									for (size_t adimension_idx = 0; adimension_idx < H_DIM_SIZE; ++adimension_idx) {
											distance_t ato_multiply = vector1[adimension_idx] - vector2[adimension_idx];
											adistance += ato_multiply * ato_multiply;
									}
									pruning_distances[v_idx] += adistance;
									//pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
									//		query + offset_query, data + data_pos, H_DIM_SIZE
									//);
							}
							
							current_dimension_idx += H_DIM_SIZE;
							//GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
							pruning_threshold = best_candidate.distance * pruner.ratios[current_dimension_idx];
							// ratios[num_dimensions] <- ratio accounts for the expected contribution of remaining dimensions to the total distance.
							// return best_candidate.distance * ratios[current_dimension_idx];
							// Basically, for the current dimension, speculatively decrease treshold (make it more tough)

							// Basically, upate the pruning threshold with the non pruned vectors (and update which ones are pruned)
							n_vectors_not_pruned = 0;
							for (size_t vector_idx = 0; vector_idx < cur_n_vectors_not_pruned; ++vector_idx) {
									pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
									n_vectors_not_pruned +=
											pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
							}
					}
				}
        // GO THROUGH THE REST IN THE VERTICAL
        while (n_vectors_not_pruned && current_vertical_dimension < pdx_data.num_vertical_dimensions
        ) {
            cur_n_vectors_not_pruned = n_vectors_not_pruned;
            // !We have the data also in the Horizontal layout, so we go till the end
            size_t dimensions_left = pdx_data.num_vertical_dimensions - current_vertical_dimension;
            size_t offset_query = current_vertical_dimension;
            //for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
            //    size_t v_idx = pruning_positions[vector_idx];
            //    auto data_pos = aux_vertical_dimensions_in_horizontal_layout +
            //                    (v_idx * pdx_data.num_vertical_dimensions) +
            //                    current_vertical_dimension;
            //    __builtin_prefetch(data_pos, 0, 1);
            //}
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                auto data_pos = aux_vertical_dimensions_in_horizontal_layout +
                                (v_idx * pdx_data.num_vertical_dimensions) +
                                current_vertical_dimension;
                pruning_distances[v_idx] += DistanceComputer<alpha, Q>::Horizontal(
                    query + offset_query, data_pos, dimensions_left
                );
            }
            current_dimension_idx = pdx_data.num_dimensions;
            current_vertical_dimension = pdx_data.num_vertical_dimensions;
            //assert( current_dimension_idx == current_vertical_dimension + current_horizontal_dimension);
            //GetPruningThreshold<Q>(best_candidate, pruning_threshold, current_dimension_idx);
						pruning_threshold = best_candidate.distance * pruner.ratios[current_dimension_idx];
            // EvaluatePruningPredicateOnPositionsArray<Q>(
            //     cur_n_vectors_not_pruned,
            //     n_vectors_not_pruned,
            //     pruning_positions,
            //     pruning_threshold,
            //     pruning_distances
            // );
							n_vectors_not_pruned = 0;
							for (size_t vector_idx = 0; vector_idx < cur_n_vectors_not_pruned; ++vector_idx) {
									pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
									n_vectors_not_pruned +=
											pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
							}
            if (current_dimension_idx == pdx_data.num_dimensions)
                break;
        }
    }

    /**
     * @brief Updates the best candidate from remaining non-pruned vectors.
     */
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

    /**
     * @brief Converts distances back to original domain (for u8 quantization).
     */
    SKM_NO_INLINE
    void BuildResultSet(KNNCandidate_t& best_candidate) {
        // We return distances in the original domain
        if constexpr (q == Quantization::u8) {
            float inverse_scale_factor = 1.0f / pdx_data.scale_factor;
            inverse_scale_factor = inverse_scale_factor * inverse_scale_factor;
            best_candidate.distance = best_candidate.distance * inverse_scale_factor;
        }
    }

  public:
    /**
     * @brief Finds the nearest neighbor using partial distances from BLAS computation.
     *
     * This variant skips the warmup phase and directly uses pre-computed partial
     * distances (e.g., from a BLAS matrix multiplication). Used in the hybrid
     * BLAS+PDX approach where BLAS computes the first partial_d dimensions.
     *
     * @param query Query vector (in rotated space)
     * @param prev_pruning_threshold Initial distance threshold
     * @param prev_top_1 Index of previous best candidate
     * @param partial_pruning_distances Pre-computed partial distances (updated in place)
     * @param computed_distance_until Number of dimensions already computed
     * @param start_cluster First cluster index to search
     * @param end_cluster One past the last cluster index to search
     * @param initial_not_pruned_accum Optional accumulator for not-pruned statistics
     * @return Best candidate found (index and distance)
     */
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
                cluster.aux_vertical_dimensions_in_horizontal_layout,
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

    SKM_NO_INLINE
    KNNCandidate_t GPUTop1PartialSearchWithThresholdAndPartialDistances(
        const float* SKM_RESTRICT query,
        const float prev_pruning_threshold,
        const uint32_t prev_top_1,
        DISTANCES_TYPE* partial_pruning_distances, // all distances_buf
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
            GPUPrune(
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
                cluster.aux_vertical_dimensions_in_horizontal_layout,
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
