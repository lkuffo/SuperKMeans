#pragma once
#ifndef SKMEANS_PDXEARCH_HPP
#define SKMEANS_PDXEARCH_HPP

#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_set>
// #include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/pdx/index_base/pdx_ivf.hpp"
#include "superkmeans/pdx/index_base/pdx_ivf2.hpp"
#include "superkmeans/pdx/pruners/adsampling.hpp"
#include "superkmeans/pdx/pruners/bond.hpp"
#include "superkmeans/pdx/utils.h"

namespace skmeans {

/******************************************************************
 * PDXearch
 * Implements our algorithm for vertical pruning
 ******************************************************************/
template <
    Quantization q = f32,
    class Index = IndexPDXIVF<q>,
    DistanceFunction alpha = l2,
    class Pruner = ADSamplingPruner<q>>
class PDXearch {
  public:
    using DISTANCES_TYPE = skmeans_distance_t<q>;
    using DATA_TYPE = skmeans_value_t<q>;
    using INDEX_TYPE = Index;
    using CLUSTER_TYPE = Cluster<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;

    Pruner pruner;
    INDEX_TYPE& pdx_data;

    PDXearch(INDEX_TYPE& data_index, Pruner& pruner) : pruner(pruner), pdx_data(data_index) {
        indices_dimensions.resize(pdx_data.num_dimensions);
        std::iota(indices_dimensions.begin(), indices_dimensions.end(), 0);
        clusters_indices.resize(pdx_data.num_clusters);
        cluster_offsets.resize(pdx_data.num_clusters);
        for (size_t i = 0; i < pdx_data.num_clusters; ++i) {
            cluster_offsets[i] = total_embeddings;
            total_embeddings += pdx_data.clusters[i].num_embeddings;
        }
        if constexpr (std::is_same_v<Pruner, BondPruner<q>>) {
            pdx_data.num_horizontal_dimensions = 0;
            pdx_data.num_vertical_dimensions = pdx_data.num_dimensions;
            is_adsampling = false;
        }
        if constexpr (std::is_same_v<Pruner, ADSamplingPruner<q>>) {
            is_adsampling = true;
        }
    }

    void PrintPrunedPositions() {
        std::vector<std::pair<uint32_t, int>> elems(
            pruned_positions.begin(), pruned_positions.end()
        );
        std::sort(elems.begin(), elems.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::cout << "{\n";
        for (const auto& [key, value] : elems) {
            std::cout << "  " << key << ": " << value << ",\n";
        }
        std::cout << "}\n";
        pruned_positions.clear();
    }

    void SetNProbe(size_t nprobe) { ivf_nprobe = nprobe; }

    TicToc end_to_end_clock = TicToc();

    void ResetClocks() { end_to_end_clock.Reset(); }

  protected:
    float selectivity_threshold = 0.80;
    size_t ivf_nprobe = 0;
    uint32_t is_adsampling = false;
    size_t total_embeddings = 0;

    std::vector<uint32_t> indices_dimensions;  // TODO: Thread local
    std::vector<uint32_t> clusters_indices;    // TODO: Thread local
    std::vector<uint32_t> clusters_indices_l0; // TODO: Thread local
    std::vector<size_t> cluster_offsets;

    static inline std::unordered_map<uint32_t, uint32_t> pruned_positions{};

    // For pruning we do not use tight loops of 64. We know that tight loops bring benefits
    // to the distance kernels (40% faster), however doing so + PRUNING in the tight block of 64
    // slightly reduces the performance of PDXearch. We are still investigating why.
    static constexpr uint16_t PDX_VECTOR_SIZE = 64;

    template <Quantization Q = q>
    void ResetPruningDistances(size_t n_vectors, skmeans_distance_t<Q>* pruning_distances) {
        memset((void*) pruning_distances, 0, n_vectors * sizeof(skmeans_distance_t<Q>));
    }

    template <Quantization Q = q>
    void ResetDistancesVectorized(skmeans_distance_t<Q>* distances) {
        memset((void*) distances, 0, PDX_VECTOR_SIZE * sizeof(skmeans_distance_t<Q>));
    }

    // The pruning threshold by default is the top of the heap
    template <Quantization Q = q>
    void GetPruningThreshold(
        uint32_t k,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap,
        skmeans_distance_t<Q>& pruning_threshold,
        uint32_t current_dimension_idx
    ) {
        pruning_threshold = pruner.template GetPruningThreshold<Q>(k, heap, current_dimension_idx);
    };

    template <Quantization Q = q>
    void EvaluatePruningPredicateScalar(
        uint32_t& n_pruned,
        size_t n_vectors,
        skmeans_distance_t<Q>* pruning_distances,
        const skmeans_distance_t<Q> pruning_threshold
    ) {
        for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    template <Quantization Q = q>
    void EvaluatePruningPredicateOnPositionsArray(
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
    };

    template <Quantization Q = q>
    void EvaluatePruningPredicateVectorized(
        uint32_t& n_pruned,
        skmeans_distance_t<Q> pruning_threshold,
        skmeans_distance_t<Q>* pruning_distances
    ) {
        for (size_t vector_idx = 0; vector_idx < PDX_VECTOR_SIZE; ++vector_idx) {
            n_pruned += pruning_distances[vector_idx] >= pruning_threshold;
        }
    };

    template <Quantization Q = q>
    void InitPositionsArray(
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
    };

    static void GetClustersAccessOrderIVF(
        const float* SKM_RESTRICT query,
        const INDEX_TYPE& data,
        size_t nprobe,
        std::vector<uint32_t>& clusters_indices
    ) {
        std::vector<float> distances_to_centroids;
        distances_to_centroids.resize(data.num_clusters);
        for (size_t cluster_idx = 0; cluster_idx < data.num_clusters; cluster_idx++) {
            distances_to_centroids[cluster_idx] = DistanceComputer<l2, f32>::Horizontal(
                query, data.centroids + cluster_idx * data.num_dimensions, data.num_dimensions
            );
        }
        clusters_indices.resize(data.num_clusters);
        std::iota(clusters_indices.begin(), clusters_indices.end(), 0);
        std::partial_sort(
            clusters_indices.begin(),
            clusters_indices.begin() + nprobe,
            clusters_indices.end(),
            [&distances_to_centroids](size_t i1, size_t i2) {
                return distances_to_centroids[i1] < distances_to_centroids[i2];
            }
        );
    }

    // On the first bucket, we do a full scan (we do not prune vectors)
    template <Quantization Q = q>
    void Start(
        const skmeans_value_t<Q>* SKM_RESTRICT query,
        const skmeans_value_t<Q>* data,
        const size_t n_vectors,
        uint32_t k,
        const uint32_t* vector_indices,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q>* pruning_distances,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap
    ) {
        ResetPruningDistances<Q>(n_vectors, pruning_distances);
        DistanceComputer<alpha, Q>::Vertical(
            query,
            data,
            n_vectors,
            n_vectors,
            0,
            pdx_data.num_vertical_dimensions,
            pruning_distances,
            pruning_positions
        );
        for (size_t horizontal_dimension = 0;
             horizontal_dimension < pdx_data.num_horizontal_dimensions;
             horizontal_dimension += H_DIM_SIZE) {
            for (size_t vector_idx = 0; vector_idx < n_vectors; vector_idx++) {
                size_t data_pos = (pdx_data.num_vertical_dimensions * n_vectors) +
                                  (horizontal_dimension * n_vectors) + (vector_idx * H_DIM_SIZE);
                pruning_distances[vector_idx] += DistanceComputer<alpha, Q>::Horizontal(
                    query + pdx_data.num_vertical_dimensions + horizontal_dimension,
                    data + data_pos,
                    H_DIM_SIZE
                );
            }
        }
        size_t max_possible_k = std::min(
            (size_t) k - heap.size(),
            n_vectors
        ); // Note: Start() should not be called if heap.size() >= k
        std::vector<size_t> indices_sorted;
        indices_sorted.resize(n_vectors);
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::partial_sort(
            indices_sorted.begin(),
            indices_sorted.begin() + max_possible_k,
            indices_sorted.end(),
            [pruning_distances](size_t i1, size_t i2) {
                return pruning_distances[i1] < pruning_distances[i2];
            }
        );
        // insert first k results into the heap
        for (size_t idx = 0; idx < max_possible_k; ++idx) {
            auto embedding = KNNCandidate<Q>{};
            size_t index = indices_sorted[idx];
            embedding.index = vector_indices[index];
            embedding.distance = pruning_distances[index];
            heap.push(embedding);
        }
    }

    // On the warmup phase, we keep scanning dimensions until the amount of not-yet pruned vectors
    // is low
    template <Quantization Q = q>
    void Warmup(
        const skmeans_value_t<Q>* SKM_RESTRICT query,
        const skmeans_value_t<Q>* SKM_RESTRICT data,
        const size_t n_vectors,
        uint32_t k,
        float tuples_threshold,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q>* pruning_distances,
        skmeans_distance_t<Q>& pruning_threshold,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap,
        uint32_t& current_dimension_idx
    ) {
        current_dimension_idx = 0;
        thread_local size_t cur_subgrouping_size_idx = 0;
        cur_subgrouping_size_idx = 0;
        size_t tuples_needed_to_exit = std::ceil(1.0 * tuples_threshold * n_vectors);
        ResetPruningDistances<Q>(n_vectors, pruning_distances);
        uint32_t n_tuples_to_prune = 0;
        if (!is_adsampling)
            GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
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
            if (is_adsampling)
                GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
            n_tuples_to_prune = 0;
            EvaluatePruningPredicateScalar<Q>(
                n_tuples_to_prune, n_vectors, pruning_distances, pruning_threshold
            );
        }
    }

    // We scan only the not-yet pruned vectors
    template <Quantization Q = q>
    void Prune(
        const skmeans_value_t<Q>* SKM_RESTRICT query,
        const skmeans_value_t<Q>* SKM_RESTRICT data,
        const size_t n_vectors,
        uint32_t k,
        uint32_t* pruning_positions,
        skmeans_distance_t<Q>* pruning_distances,
        skmeans_distance_t<Q>& pruning_threshold,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap,
        size_t& n_vectors_not_pruned,
        uint32_t& current_dimension_idx,
        const skmeans_value_t<Q>* SKM_RESTRICT aux_data = nullptr
    ) {
        GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
        InitPositionsArray<Q>(
            n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances
        );
        // std::cout << "mmmm: " << n_vectors_not_pruned << "\n";
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
            if (is_adsampling)
                GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
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
            if (aux_data == nullptr) {
                //std::cout << "huh?" << "\n";
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
                // std::cout << "CVD: " << current_vertical_dimension << "\n";
                // std::cout << "Dims left: " << dimensions_left << "\n";
                // std::cout << "NNP: " << n_vectors_not_pruned << "\n";
                // std::cout << "PT: " << pruning_threshold << "\n";
                size_t offset_query = current_vertical_dimension;
                for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                    size_t v_idx = pruning_positions[vector_idx];
                    auto data_pos = aux_data + (v_idx * pdx_data.num_vertical_dimensions) +
                                    current_vertical_dimension;
                    __builtin_prefetch(data_pos, 0, 2);
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
            if (is_adsampling)
                GetPruningThreshold<Q>(k, heap, pruning_threshold, current_dimension_idx);
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

    template <bool IS_PRUNING = false, Quantization Q = q>
    void MergeIntoHeap(
        const uint32_t* vector_indices,
        size_t n_vectors,
        uint32_t k,
        const uint32_t* pruning_positions,
        skmeans_distance_t<Q>* pruning_distances,
        skmeans_distance_t<Q>* distances,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap
    ) {
        for (size_t position_idx = 0; position_idx < n_vectors; ++position_idx) {
            size_t index = position_idx;
            // DISTANCES_TYPE current_distance;
            float current_distance;
            if constexpr (IS_PRUNING) {
                index = pruning_positions[position_idx];
                current_distance = pruning_distances[index];
            } else {
                current_distance = distances[index];
            }
            if (heap.size() < k || current_distance < heap.top().distance) {
                KNNCandidate<Q> embedding{};
                embedding.distance = current_distance;
                embedding.index = vector_indices[index];
                if (heap.size() >= k) {
                    heap.pop();
                }
                heap.push(embedding);
            }
        }
    }

    std::vector<KNNCandidate_t> BuildResultSet(
        uint32_t k,
        std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>& best_k
    ) {
        size_t result_set_size = std::min(best_k.size(), (size_t) k);
        std::vector<KNNCandidate_t> result;
        result.resize(result_set_size);
        // We return distances in the original domain (do we need it?)
        float inverse_scale_factor;
        if constexpr (q == u8) {
            inverse_scale_factor = 1.0f / pdx_data.scale_factor;
            inverse_scale_factor = inverse_scale_factor * inverse_scale_factor;
        }
        for (int i = result_set_size - 1; i >= 0; --i) {
            const KNNCandidate_t& embedding = best_k.top();
            if constexpr (q == u8) {
                result[i].distance = embedding.distance * inverse_scale_factor;
            } else if constexpr (q == f32) {
                result[i].distance = embedding.distance;
            }
            result[i].index = embedding.index;
            best_k.pop();
        }
        return result;
    }

    void BuildResultSetCentroids(
        uint32_t k,
        std::priority_queue<
            KNNCandidate<f32>,
            std::vector<KNNCandidate<f32>>,
            VectorComparator<f32>>& best_k_centroids
    ) {
        for (int i = k - 1; i >= 0; --i) {
            const KNNCandidate<f32>& embedding = best_k_centroids.top();
            clusters_indices[i] = embedding.index;
            best_k_centroids.pop();
        }
    }

    // We store centroids using PDX in tight blocks of 64
    // TODO: Always assumes multiple of 64
    void GetL0ClustersAccessOrderPDX(const float* SKM_RESTRICT query) {
        alignas(64) thread_local skmeans_distance_t<f32> centroids_distances[PDX_VECTOR_SIZE];
        clusters_indices_l0.resize(pdx_data.num_clusters_l0);
        std::iota(clusters_indices_l0.begin(), clusters_indices_l0.end(), 0);
        float* tmp_centroids_pdx = pdx_data.centroids_pdx;
        size_t SKIPPING_SIZE = PDX_VECTOR_SIZE * pdx_data.num_dimensions;
        size_t full_blocks = std::floor(1.0 * pdx_data.num_clusters_l0 / PDX_VECTOR_SIZE);
        for (size_t centroid_idx = 0; centroid_idx < full_blocks; ++centroid_idx) {
            memset((void*) centroids_distances, 0, PDX_VECTOR_SIZE * sizeof(float));
            DistanceComputer<alpha, f32>::VerticalBlock(
                query, tmp_centroids_pdx, 0, pdx_data.num_dimensions, centroids_distances
            );
            tmp_centroids_pdx += SKIPPING_SIZE;
        }
        std::vector<size_t> indices_sorted;
        indices_sorted.resize(pdx_data.num_clusters_l0);
        std::iota(indices_sorted.begin(), indices_sorted.end(), 0);
        std::partial_sort(
            indices_sorted.begin(),
            indices_sorted.begin() + 64,
            indices_sorted.end(),
            [](size_t i1, size_t i2) { return centroids_distances[i1] < centroids_distances[i2]; }
        );
        // Sort the distance of the first N centroids to determine access order
        for (size_t idx = 0; idx < pdx_data.num_clusters_l0; ++idx) {
            clusters_indices_l0[idx] = indices_sorted[idx];
        }
    }

    void GetL1ClustersAccessOrderPDX(
        const float* SKM_RESTRICT query,
        size_t n_buckets,
        bool safe_to_prune_space = true
    ) {
        alignas(64) thread_local skmeans_distance_t<f32> pruning_distances_l0[10240];
        alignas(64) thread_local uint32_t pruning_positions_l0[10240];
        thread_local auto best_k_centroids = std::priority_queue<
            KNNCandidate<f32>,
            std::vector<KNNCandidate<f32>>,
            VectorComparator<f32>>{};
        thread_local skmeans_distance_t<f32> pruning_threshold_l0 =
            std::numeric_limits<skmeans_distance_t<f32>>::max();
        thread_local size_t current_cluster = 0;
        thread_local uint32_t current_dimension_idx = 0;
        thread_local size_t n_vectors_not_pruned = 0;
        best_k_centroids = {};
        current_cluster = 0;
        current_dimension_idx = 0;
        n_vectors_not_pruned = 0;
        size_t clusters_to_visit = pdx_data.num_clusters_l0;
        if ((n_buckets < pdx_data.num_clusters / 2) && safe_to_prune_space) {
            // We prune half of the super-clusters only if the user wants to
            // visit less than half of the available clusters
            clusters_to_visit = pdx_data.num_clusters_l0 / 2;
        }
        current_dimension_idx = 0;
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices_l0[cluster_idx];
            Cluster<f32>& cluster = pdx_data.clusters_l0[current_cluster];
            if (best_k_centroids.size() < n_buckets) {
                // The heap may not be filled with the first super-cluster
                // if the number of clusters probed is high or if the
                // number of vectors per super-cluster is small.
                Start<f32>(
                    query,
                    cluster.data,
                    cluster.num_embeddings,
                    n_buckets,
                    cluster.indices,
                    pruning_positions_l0,
                    pruning_distances_l0,
                    best_k_centroids
                );
                continue;
            }
            Warmup<f32>(
                query,
                cluster.data,
                cluster.num_embeddings,
                n_buckets,
                selectivity_threshold,
                pruning_positions_l0,
                pruning_distances_l0,
                pruning_threshold_l0,
                best_k_centroids,
                current_dimension_idx
            );
            Prune<f32>(
                query,
                cluster.data,
                cluster.num_embeddings,
                n_buckets,
                pruning_positions_l0,
                pruning_distances_l0,
                pruning_threshold_l0,
                best_k_centroids,
                n_vectors_not_pruned,
                current_dimension_idx
            );
            if (n_vectors_not_pruned) {
                MergeIntoHeap<true, f32>(
                    cluster.indices,
                    n_vectors_not_pruned,
                    n_buckets,
                    pruning_positions_l0,
                    pruning_distances_l0,
                    nullptr,
                    best_k_centroids
                );
            }
        }
        // Rare case in which half of the space is not enough to fill the best_k_centroids heap
        if (best_k_centroids.size() < n_buckets) {
            // From 32 to 64 (in the default case)
            for (size_t cluster_idx = clusters_to_visit; cluster_idx < pdx_data.num_clusters_l0;
                 ++cluster_idx) {
                current_cluster = clusters_indices_l0[cluster_idx];
                Cluster<f32>& cluster = pdx_data.clusters_l0[current_cluster];
                Start<f32>(
                    query,
                    cluster.data,
                    cluster.num_embeddings,
                    n_buckets,
                    cluster.indices,
                    pruning_positions_l0,
                    pruning_distances_l0,
                    best_k_centroids
                );
                if (best_k_centroids.size() == n_buckets) {
                    break;
                }
            }
        }
        BuildResultSetCentroids(n_buckets, best_k_centroids);
    }

    void GetClustersAccessOrderRandom() {
        std::iota(clusters_indices.begin(), clusters_indices.end(), 0);
    }

  public:
    /******************************************************************
     * Search methods
     ******************************************************************/
    std::vector<KNNCandidate_t> Search(const float* SKM_RESTRICT raw_query, uint32_t k) {
        alignas(64) thread_local uint32_t pruning_positions[10240];
        alignas(64) thread_local DISTANCES_TYPE distances[PDX_VECTOR_SIZE];
        alignas(64) thread_local DISTANCES_TYPE pruning_distances[10240];
        thread_local size_t current_cluster = 0;
        thread_local DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        thread_local auto best_k =
            std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        thread_local size_t n_vectors_not_pruned = 0;
        thread_local uint32_t current_dimension_idx = 0;
        pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        best_k = {};
        current_dimension_idx = 0;
        n_vectors_not_pruned = 0;
        current_cluster = 0;
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif

        alignas(64) float query[pdx_data.num_dimensions];
        if (!pdx_data.is_normalized) {
            pruner.PreprocessQuery(raw_query, query);
        } else {
            alignas(64) float normalized_query[pdx_data.num_dimensions];
            pruner.PreprocessQuery(normalized_query, query);
        }
        size_t clusters_to_visit = (ivf_nprobe == 0 || ivf_nprobe > pdx_data.num_clusters)
                                       ? pdx_data.num_clusters
                                       : ivf_nprobe;
        if constexpr (std::is_same_v<Index, IndexPDXIVF2<q>>) {
            // Multilevel access
            GetL0ClustersAccessOrderPDX(query);
            GetL1ClustersAccessOrderPDX(query, clusters_to_visit);
        } else {
            if (pdx_data.is_ivf) {
                GetClustersAccessOrderIVF(query, pdx_data, clusters_to_visit, clusters_indices);
            } else {
                // If there is no index, we just access the clusters in order
                GetClustersAccessOrderRandom();
            }
        }
        // PDXearch core
        current_dimension_idx = 0;
        DATA_TYPE* prepared_query = query;
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices[cluster_idx];
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            if (best_k.size() < k) {
                // We cannot prune until we fill the heap
                Start(
                    prepared_query,
                    cluster.data,
                    cluster.num_embeddings,
                    k,
                    cluster.indices,
                    pruning_positions,
                    pruning_distances,
                    best_k
                );
                continue;
            }
            Warmup(
                prepared_query,
                cluster.data,
                cluster.num_embeddings,
                k,
                selectivity_threshold,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                current_dimension_idx
            );
            Prune(
                prepared_query,
                cluster.data,
                cluster.num_embeddings,
                k,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                n_vectors_not_pruned,
                current_dimension_idx
            );
            if (n_vectors_not_pruned) {
                MergeIntoHeap<true>(
                    cluster.indices,
                    n_vectors_not_pruned,
                    k,
                    pruning_positions,
                    pruning_distances,
                    nullptr,
                    best_k
                );
            }
        }
        std::vector<KNNCandidate_t> result = BuildResultSet(k, best_k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

    std::vector<KNNCandidate_t> Top1Search(const float* SKM_RESTRICT query) {
        alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];
        // alignas(64) thread_local DISTANCES_TYPE distances[PDX_VECTOR_SIZE];
        alignas(64) thread_local DISTANCES_TYPE pruning_distances[PDX_VECTOR_SIZE];
        thread_local DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        thread_local auto best_k =
            std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        thread_local size_t n_vectors_not_pruned = 0;
        thread_local uint32_t current_dimension_idx = 0;
        thread_local size_t current_cluster = 0;
        pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        best_k = {};
        current_dimension_idx = 0;
        n_vectors_not_pruned = 0;
        current_cluster = 0;
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        constexpr uint32_t k = 1;
        size_t clusters_to_visit = pdx_data.num_clusters;
        if constexpr (std::is_same_v<Index, IndexPDXIVF2<q>>) {
            // Multilevel access
            GetL0ClustersAccessOrderPDX(query);
            GetL1ClustersAccessOrderPDX(query, clusters_to_visit);
        } else {
            if (pdx_data.is_ivf) {
                GetClustersAccessOrderIVF(query, pdx_data, clusters_to_visit, clusters_indices);
            } else {
                // If there is no index, we just access the clusters in order
                GetClustersAccessOrderRandom();
            }
        }
        // PDXearch core
        current_dimension_idx = 0;
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_cluster = clusters_indices[cluster_idx];
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            if (best_k.size() < k) {
                // We cannot prune until we fill the heap
                Start(
                    query,
                    cluster.data,
                    cluster.num_embeddings,
                    k,
                    cluster.indices,
                    pruning_positions,
                    pruning_distances,
                    best_k
                );
                continue;
            }
            Warmup(
                query,
                cluster.data,
                cluster.num_embeddings,
                k,
                selectivity_threshold,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                current_dimension_idx
            );
            Prune(
                query,
                cluster.data,
                cluster.num_embeddings,
                k,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                n_vectors_not_pruned,
                current_dimension_idx
            );
            if (n_vectors_not_pruned) {
                MergeIntoHeap<true>(
                    cluster.indices,
                    n_vectors_not_pruned,
                    k,
                    pruning_positions,
                    pruning_distances,
                    nullptr,
                    best_k
                );
            }
        }
        std::vector<KNNCandidate_t> result = BuildResultSet(k, best_k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

    std::vector<KNNCandidate_t> Top1SearchWithThreshold(
        const float* SKM_RESTRICT query,
        const float prev_pruning_threshold,
        const uint32_t prev_top_1,
        const size_t vector_id,
        uint32_t& pruned_at_accum,
        const size_t start_cluster,
        const size_t end_cluster
    ) {
        alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];
        // alignas(64) thread_local DISTANCES_TYPE distances[PDX_VECTOR_SIZE];
        alignas(64) thread_local DISTANCES_TYPE pruning_distances[PDX_VECTOR_SIZE];
        DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        thread_local auto best_k =
            std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        thread_local size_t n_vectors_not_pruned = 0;
        thread_local uint32_t current_dimension_idx = 0;
        thread_local size_t current_cluster = 0;

        best_k = {};
        // size_t pruned_at = 0;

        // Setup previous top1
        pruning_threshold = prev_pruning_threshold;
        auto top_embedding = KNNCandidate<q>{};
        top_embedding.index = prev_top_1;
        top_embedding.distance = prev_pruning_threshold;
        best_k.push(top_embedding);

        current_dimension_idx = 0;
        n_vectors_not_pruned = 0;
        current_cluster = 0;
#ifdef BENCHMARK_TIME
        this->ResetClocks();
        this->end_to_end_clock.Tic();
#endif
        constexpr uint32_t k = 1;
        size_t clusters_to_visit = pdx_data.num_clusters;
        // PDXearch core
        current_dimension_idx = 0;
        for (size_t cluster_idx = start_cluster; cluster_idx < end_cluster; ++cluster_idx) {
            current_cluster = cluster_idx;
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            Warmup(
                query,
                cluster.data,
                cluster.num_embeddings,
                k,
                selectivity_threshold,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                current_dimension_idx
            );
            pruned_at_accum += current_dimension_idx;
            Prune(
                query,
                cluster.data,
                cluster.num_embeddings,
                k,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                n_vectors_not_pruned,
                current_dimension_idx
            );
            if (n_vectors_not_pruned) {
                MergeIntoHeap<true>(
                    cluster.indices,
                    n_vectors_not_pruned,
                    k,
                    pruning_positions,
                    pruning_distances,
                    nullptr,
                    best_k
                );
            }
        }
        // float pruned_at_avg = (1.0 * pruned_at) / clusters_to_visit;
        // uint32_t group = static_cast<uint32_t>(std::ceil(pruned_at_avg / 16.0f) * 16.0f);
        // pruned_positions[group] += 1;
        // if (vector_id % 10000 == 0) {
        //     std::cout << "Vector " << vector_id << " | "
        //               << "Pruned At Avg.: " << (1.0 * pruned_at) / clusters_to_visit <<
        //               std::endl;
        // }
        std::vector<KNNCandidate_t> result = BuildResultSet(k, best_k);
#ifdef BENCHMARK_TIME
        this->end_to_end_clock.Toc();
#endif
        return result;
    }

    std::vector<KNNCandidate_t> Top1SearchWithThresholdAndPartialDistances(
        const float* SKM_RESTRICT query,
        const float prev_pruning_threshold,
        const uint32_t prev_top_1,
        const size_t vector_id,
        DISTANCES_TYPE* partial_pruning_distances,
        const uint32_t computed_distance_until
    ) {
        alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];
        DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        thread_local auto best_k =
            std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        thread_local size_t n_vectors_not_pruned = 0;
        thread_local uint32_t current_dimension_idx = 0;
        thread_local size_t current_cluster = 0;

        best_k = {};
        size_t pruned_at = 0;

        // Setup previous top1
        pruning_threshold = prev_pruning_threshold;
        auto top_embedding = KNNCandidate<q>{};
        top_embedding.index = prev_top_1;
        top_embedding.distance = prev_pruning_threshold;
        best_k.push(top_embedding);

        current_dimension_idx = computed_distance_until;
        n_vectors_not_pruned = 0;
        current_cluster = 0;
        constexpr uint32_t k = 1;
        size_t clusters_to_visit = pdx_data.num_clusters;
        // PDXearch core
        size_t data_offset = 0;
        for (size_t cluster_idx = 0; cluster_idx < clusters_to_visit; ++cluster_idx) {
            current_dimension_idx = computed_distance_until;
            auto pruning_distances = partial_pruning_distances + data_offset;
            current_cluster = cluster_idx;
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            data_offset += cluster.num_embeddings;
            Prune(
                query,
                cluster.data,
                cluster.num_embeddings,
                k,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                n_vectors_not_pruned,
                current_dimension_idx
            );
            if (n_vectors_not_pruned) {
                MergeIntoHeap<true>(
                    cluster.indices,
                    n_vectors_not_pruned,
                    k,
                    pruning_positions,
                    pruning_distances,
                    nullptr,
                    best_k
                );
            }
        }
        // if (vector_id % 10000 == 0) {
        //     std::cout << "Vector " << vector_id << " | "
        //               << "Pruned At Avg.: " << (1.0 * pruned_at) / clusters_to_visit <<
        //               std::endl;
        // }
        std::vector<KNNCandidate_t> result = BuildResultSet(k, best_k);
        return result;
    }

    std::vector<KNNCandidate_t> Top1PartialSearchWithThresholdAndPartialDistances(
        const float* SKM_RESTRICT query,
        const float prev_pruning_threshold,
        const uint32_t prev_top_1,
        const size_t vector_id,
        DISTANCES_TYPE* partial_pruning_distances,
        const uint32_t computed_distance_until,
        const size_t start_cluster,
        const size_t end_cluster
    ) {
        alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];
        DISTANCES_TYPE pruning_threshold = std::numeric_limits<DISTANCES_TYPE>::max();
        thread_local auto best_k =
            std::priority_queue<KNNCandidate_t, std::vector<KNNCandidate_t>, VectorComparator_t>{};
        thread_local size_t n_vectors_not_pruned = 0;
        thread_local uint32_t current_dimension_idx = 0;
        thread_local size_t current_cluster = 0;

        best_k = {};
        size_t pruned_at = 0;

        // Setup previous top1
        pruning_threshold = prev_pruning_threshold;
        auto top_embedding = KNNCandidate<q>{};
        top_embedding.index = prev_top_1;
        top_embedding.distance = prev_pruning_threshold;
        best_k.push(top_embedding);

        current_dimension_idx = computed_distance_until;
        n_vectors_not_pruned = 0;
        current_cluster = 0;
        constexpr uint32_t k = 1;
        // PDXearch core
        size_t data_offset = 0;
        for (size_t cluster_idx = start_cluster; cluster_idx < end_cluster; ++cluster_idx) {
            current_dimension_idx = computed_distance_until;
            // std::cout << computed_distance_until << std::endl;
            auto pruning_distances = partial_pruning_distances + data_offset;
            current_cluster = cluster_idx;
            CLUSTER_TYPE& cluster = pdx_data.clusters[current_cluster];
            data_offset += cluster.num_embeddings;
            Prune(
                query,
                cluster.data,
                cluster.num_embeddings,
                k,
                pruning_positions,
                pruning_distances,
                pruning_threshold,
                best_k,
                n_vectors_not_pruned,
                current_dimension_idx,
                cluster.aux_hor_data
            );
            if (n_vectors_not_pruned) {
                MergeIntoHeap<true>(
                    cluster.indices,
                    n_vectors_not_pruned,
                    k,
                    pruning_positions,
                    pruning_distances,
                    nullptr,
                    best_k
                );
            }
        }
        // if (vector_id % 10000 == 0) {
        //     std::cout << "Vector " << vector_id << " | "
        //               << "Pruned At Avg.: " << (1.0 * pruned_at) / clusters_to_visit <<
        //               std::endl;
        // }
        std::vector<KNNCandidate_t> result = BuildResultSet(k, best_k);
        return result;
    }
};

} // namespace skmeans

#endif // SKMEANS_PDXEARCH_HPP
