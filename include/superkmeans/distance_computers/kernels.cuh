#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include <Eigen/Eigen/Dense>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
// #include "superkmeans/profiler.h"

namespace skmeans {
namespace kernels {

using distance_t = skmeans_distance_t<Quantization::f32>;
using data_t = skmeans_value_t<Quantization::f32>;
using norms_t = skmeans_value_t<Quantization::f32>;
using knn_candidate_t = KNNCandidate<Quantization::f32>;
using layout_t = PDXLayout<Quantization::f32, DistanceFunction::l2>;

void first_blas(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    distance_t* out_distances,
    uint32_t* out_knn,
    const cudaStream_t stream
);

void norms(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    const cudaStream_t stream
);

template <uint32_t NUM_DIMENSIONS>
static skmeans_distance_t<Quantization::f32> GPUDistanceHorizontalFixedDimensions(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector1,
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector2
) {
    skmeans_distance_t<Quantization::f32> distance = 0.0;
    // DISTANCE calculation Do this with 32 threads
    for (size_t dimension_idx = 0; dimension_idx < NUM_DIMENSIONS; ++dimension_idx) {
        skmeans_distance_t<Quantization::f32> to_multiply =
            vector1[dimension_idx] - vector2[dimension_idx];
        distance += to_multiply * to_multiply;
    }
    return distance;
};

static skmeans_distance_t<Quantization::f32> GPUDistanceHorizontal(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector1,
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector2,
    const uint32_t num_dimensions
) {
    skmeans_distance_t<Quantization::f32> distance = 0.0;
    // DISTANCE calculation Do this with 32 threads
    for (size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
        skmeans_distance_t<Quantization::f32> to_multiply =
            vector1[dimension_idx] - vector2[dimension_idx];
        distance += to_multiply * to_multiply;
    }
    return distance;
};

static void GPUInitializeNotPrunedVectors(
    size_t& n_vectors_not_pruned,
    const skmeans_distance_t<Quantization::f32> pruning_threshold,
    uint32_t* pruning_positions,
    skmeans_distance_t<Quantization::f32>* pruning_distances,
    const size_t n_vectors
) {
    n_vectors_not_pruned = 0;
    // Try and do this collaboratively with a warp
    // Can do a collaborative load, and then supply first thread in warp with values by downshifting
    // them with shuffle_sync
    for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
        pruning_positions[n_vectors_not_pruned] = vector_idx;
        n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
    }
}

static void GPUUpdateNotPrunedVectors(
    size_t& n_vectors_not_pruned,
    const skmeans_distance_t<Quantization::f32> pruning_threshold,
    uint32_t* pruning_positions,
    skmeans_distance_t<Quantization::f32>* pruning_distances
) {
    auto new_n_vectors_not_pruned = 0;
    // Try and do this collaboratively with a warp
    // Can do a collaborative load, and then supply first thread in warp with values by downshifting
    // them with shuffle_sync
    for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; ++vector_idx) {
        pruning_positions[new_n_vectors_not_pruned] = pruning_positions[vector_idx];
        new_n_vectors_not_pruned +=
            pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
    }
    n_vectors_not_pruned = new_n_vectors_not_pruned;
}

static void GPUSelectBestCandidate(
    KNNCandidate<Quantization::f32>& best_candidate,
    const size_t n_vectors_not_pruned,
    const uint32_t* SKM_RESTRICT vector_indices,
    const uint32_t* SKM_RESTRICT pruning_positions,
    const skmeans_distance_t<Quantization::f32>* SKM_RESTRICT pruning_distances
) {
    // TODO For all kernels, size_t is not necessary here
    // TODO: do this collaboratively
    for (size_t position_idx = 0; position_idx < n_vectors_not_pruned; ++position_idx) {
        size_t index = pruning_positions[position_idx];
        auto current_distance = pruning_distances[index];
        if (current_distance < best_candidate.distance) {
            best_candidate.distance = current_distance;
            best_candidate.index = vector_indices[index];
        }
    }
}

class ConstantPruneData {
  public:
    size_t num_dimensions;
    size_t num_horizontal_dimensions;
    size_t num_vertical_dimensions;
    std::vector<float> ratios;

    ConstantPruneData(const layout_t& pdx_centroids) {
        num_dimensions = pdx_centroids.searcher->pdx_data.num_dimensions;
        num_horizontal_dimensions = pdx_centroids.searcher->pdx_data.num_horizontal_dimensions;
        num_vertical_dimensions = pdx_centroids.searcher->pdx_data.num_vertical_dimensions;
        ratios = pdx_centroids.searcher->pruner.ratios;
    }
};

class ClusterPruneData {
  public:
    skmeans_value_t<Quantization::f32>* data;
    size_t n_vectors;
    uint32_t* vector_indices;
    skmeans_value_t<Quantization::f32>* aux_vertical_dimensions_in_horizontal_layout;

    ClusterPruneData(Cluster<Quantization::f32> y_batch_cluster) {
        data = y_batch_cluster.data;
        n_vectors = y_batch_cluster.num_embeddings;
        vector_indices = y_batch_cluster.indices;
        aux_vertical_dimensions_in_horizontal_layout =
            y_batch_cluster.aux_vertical_dimensions_in_horizontal_layout;
    }
};

static void GPUPrune(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT query,
    // const size_t y_batch,
    const ConstantPruneData constant_prune_data,
    const ClusterPruneData cluster_prune_data,
    skmeans_distance_t<Quantization::f32>* pruning_distances, // all_distances_buf
    KNNCandidate<Quantization::f32>& best_candidate, // initially prev centroid, then updated
    uint32_t current_dimension_idx,                  //??
    size_t& initial_not_pruned_accum
) {
    // TODO Make shared
    alignas(64) thread_local uint32_t pruning_positions[PDX_VECTOR_SIZE];

    const auto prev_best_candidate_distance = best_candidate.distance;

    auto pruning_threshold =
        prev_best_candidate_distance * constant_prune_data.ratios[current_dimension_idx];
    size_t n_vectors_not_pruned = 0;

    GPUInitializeNotPrunedVectors(
        n_vectors_not_pruned,
        pruning_threshold,
        pruning_positions,
        pruning_distances,
        cluster_prune_data.n_vectors
    );
    // Record the initial n_vectors_not_pruned if requested
    // WARNING: Need to do this without memory write, this is expensive
    initial_not_pruned_accum += n_vectors_not_pruned;

    // Early exit if all vectors were pruned
    if (n_vectors_not_pruned == 0) {
        return;
    }

    // Early exit if the only remaining point is the one that was initially the best candidate
    if (n_vectors_not_pruned == 1 &&
        cluster_prune_data.vector_indices[pruning_positions[0]] == best_candidate.index) {
        n_vectors_not_pruned = 0;
        return;
    }

    size_t current_vertical_dimension = current_dimension_idx;
    // TODO Make sure this is otherwise compiled away
    const bool has_horizontal_dimensions = constant_prune_data.num_horizontal_dimensions > 0;
    if (has_horizontal_dimensions) {
        for (size_t current_horizontal_dimension = 0;
             current_horizontal_dimension < constant_prune_data.num_horizontal_dimensions;
             current_horizontal_dimension += H_DIM_SIZE) {
            size_t offset_data =
                (constant_prune_data.num_vertical_dimensions * cluster_prune_data.n_vectors) +
                (current_horizontal_dimension * cluster_prune_data.n_vectors);
            size_t offset_query =
                constant_prune_data.num_vertical_dimensions + current_horizontal_dimension;
            // Can do a collaborative load for pruning_positions, and then supply
            // first thread in warp with values by downshifting them with shuffle_sync
            // Can do a collaborative write for pruning_distances, by upshifting values
            // and then doing a single write
            for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
                size_t v_idx = pruning_positions[vector_idx];
                size_t data_pos = offset_data + (v_idx * H_DIM_SIZE);
                pruning_distances[v_idx] += GPUDistanceHorizontalFixedDimensions<H_DIM_SIZE>(
                    query + offset_query, cluster_prune_data.data + data_pos
                );
            }

            current_dimension_idx += H_DIM_SIZE;
            pruning_threshold =
                prev_best_candidate_distance * constant_prune_data.ratios[current_dimension_idx];
            GPUUpdateNotPrunedVectors(
                n_vectors_not_pruned, pruning_threshold, pruning_positions, pruning_distances
            );
            if (n_vectors_not_pruned == 0) {
                break;
            }
        }
    }
    // GO THROUGH THE REST IN THE VERTICAL
    if (n_vectors_not_pruned &&
        current_vertical_dimension < constant_prune_data.num_vertical_dimensions) {
        size_t dimensions_left =
            constant_prune_data.num_vertical_dimensions - current_vertical_dimension;
        size_t offset_query = current_vertical_dimension;
        // Can do a collaborative load for pruning_positions, and then supply
        // first thread in warp with values by downshifting them with shuffle_sync
        // Can do a collaborative write for pruning_distances, by upshifting values
        // and then doing a single write
        for (size_t vector_idx = 0; vector_idx < n_vectors_not_pruned; vector_idx++) {
            size_t v_idx = pruning_positions[vector_idx];
            auto data_pos = cluster_prune_data.aux_vertical_dimensions_in_horizontal_layout +
                            (v_idx * constant_prune_data.num_vertical_dimensions) +
                            current_vertical_dimension;
            pruning_distances[v_idx] +=
                GPUDistanceHorizontal(query + offset_query, data_pos, dimensions_left);
        }
        current_dimension_idx = constant_prune_data.num_dimensions;
        pruning_threshold =
            prev_best_candidate_distance * constant_prune_data.ratios[current_dimension_idx];
        GPUUpdateNotPrunedVectors(
            n_vectors_not_pruned, pruning_threshold, pruning_positions, pruning_distances
        );
    }

    if (n_vectors_not_pruned) {
        GPUSelectBestCandidate(
            best_candidate,
            n_vectors_not_pruned,
            cluster_prune_data.vector_indices,
            pruning_positions,
            pruning_distances
        );
    }
}

static void GPUSearchPDX(
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t i,
    const size_t d,
    const uint32_t partial_d,
    const data_t* SKM_RESTRICT x,
    const ConstantPruneData constant_prune_data,
    const ClusterPruneData cluster_prune_data,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    size_t* SKM_RESTRICT out_not_pruned_counts,
    float* SKM_RESTRICT all_distances_buf
) {
    // TODO For all kernels, size_t is not necessary here

#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
    for (size_t r = 0; r < batch_n_x; ++r) {
        const auto i_idx = i + r;
        auto data_p = x + (i_idx * d);

        knn_candidate_t assigned_centroid{out_knn[i_idx], out_distances[i_idx]};

        // PDXearch per vector
        auto partial_distances_p = all_distances_buf + r * batch_n_y;
        size_t local_not_pruned = 0;

        GPUPrune(
            data_p,
            constant_prune_data,
            cluster_prune_data,
            partial_distances_p,
            assigned_centroid,
            partial_d,
            local_not_pruned
        );

        out_not_pruned_counts[i_idx] += local_not_pruned;
        out_knn[i_idx] = assigned_centroid.index;
        out_distances[i_idx] = assigned_centroid.distance;
    }
}

void GPUCalculateDistanceToCurrentCentroids(
    const uint32_t n_x,
    const uint32_t n_y,
    const uint32_t d,
    const data_t* SKM_RESTRICT x,
    const data_t* SKM_RESTRICT y,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
		const cudaStream_t stream
);

} // namespace kernels
} // namespace skmeans
