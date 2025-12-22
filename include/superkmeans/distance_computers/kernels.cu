#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <system_error>
#include <utility>

#include "superkmeans/distance_computers/kernels.cuh"

namespace skmeans {
namespace kernels {

inline void check_CUDA_error(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(
            stderr, "CUDA Error: %s (%d) at %s:%d\n", cudaGetErrorString(code), code, file, line
        );
        if (abort)
            exit(code);
    }
}

#define CUDA_SAFE_CALL(ans) check_CUDA_error((ans), __FILE__, __LINE__)

const uint32_t WARP_WIDTH = 32;
const uint32_t FULL_MASK = 0xffffffff;
const uint32_t FIRST_LANE_MASK = 0x0;

struct ThreadContext {
    uint32_t thread_id;
    uint32_t warp_id;
    uint32_t lane_id;

    uint32_t block_thread_id;
    uint32_t block_warp_id;
    __device__ ThreadContext() {
        thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        warp_id = thread_id / WARP_WIDTH;
        lane_id = thread_id % WARP_WIDTH;

        block_thread_id = threadIdx.x;
        block_warp_id = block_thread_id / WARP_WIDTH;
    }

    __device__ bool is_first_thread() const { return thread_id == 0; }
    __device__ bool is_first_lane() const { return lane_id == 0; }
};

template <typename T>
__device__ T find_min(const T* data, const uint32_t size, const T max, uint32_t* min_index) {
    T min = max;

    for (int i{0}; i < size; ++i) {
        auto current = data[i];
        if (current < min) {
            *min_index = i;
            min = current;
        }
    }

    return min;
}

template <typename T>
__device__ constexpr T max_of_either(const T a, const T b) {
    if (a < b) {
        return b;
    }
    return a;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T value) {
    for (int offset = WARP_WIDTH / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }

    return value; // valid in first lane only
}

template <typename T>
__device__ __forceinline__ T warp_broadcast(T value) {
    return __shfl_sync(FULL_MASK, value, FIRST_LANE_MASK);
}

__device__ __forceinline__ uint32_t
warp_exclusive_prefix_sum_binary(const bool value, const uint32_t mask) {
    uint32_t ballot = __ballot_sync(mask, value);
    uint32_t lane = threadIdx.x & 31;

    // Keep bits of lower lanes
    uint32_t lower = ballot & ((1u << lane) - 1);
    return __popc(lower);
}

template <typename T>
__device__ __forceinline__ T
warp_exclusive_prefix_sum_binary_and_total(T& offset, const bool value, const uint32_t mask) {
    // Useful within a for loop, where each thread does one iteration (32 iterations per warp),
    // but need offset to contain the offset of highest lane
    uint32_t ballot = __ballot_sync(mask, value);
    uint32_t lane = threadIdx.x & 31;
    offset += __popc(ballot & mask);

    // Keep bits of lower lanes
    T lower_bits = static_cast<T>(ballot & ((1u << lane) - 1));
    return __popc(lower_bits);
}

struct IndexMinWarpReductionResult {
    int32_t index;
    float value;
};

__device__ IndexMinWarpReductionResult
index_min_warp_reduction(const int32_t index, const float value) {
    auto send_index = index;
    auto send_value = value;

    for (int offset = 16; offset > 0; offset /= 2) {
        auto receive_index = __shfl_down_sync(FULL_MASK, send_index, offset);
        auto receive_value = __shfl_down_sync(FULL_MASK, send_value, offset);

        if (receive_value < send_value) {
            send_value = receive_value;
            send_index = receive_index;
        }
    }

    return IndexMinWarpReductionResult{send_index, send_value};
}

__global__ void first_blas_kernel(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    distance_t* out_distances,
    uint32_t* out_knn,
    const float max
) {
    auto warp_thread_index = threadIdx.x % WARP_WIDTH;
    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    auto item_index = global_thread_index / WARP_WIDTH;

    if (batch_n_x <= item_index) {
        return;
    }

    const float norm_x_i = norms_x[item_index];
    float* row_p = all_distances_buffer + item_index * batch_n_y;

    int32_t knn_idx = 0;
    float batch_top_1 = max;
    for (uint32_t c = warp_thread_index; c < batch_n_y; c += WARP_WIDTH) {
        auto result = -2.0f * row_p[c] + norm_x_i + norms_y[c];
        if (result < batch_top_1) {
            knn_idx = c;
            batch_top_1 = result;
        }
    }

    auto result = index_min_warp_reduction(knn_idx, batch_top_1);

    if (warp_thread_index == 0) {
        knn_idx = result.index;
        batch_top_1 = result.value;
        if (batch_top_1 < out_distances[item_index]) {
            out_distances[item_index] = max_of_either<float>(0.0f, batch_top_1);
            out_knn[item_index] = j + knn_idx;
        }
    }
}

__global__ void norms_kernel(
    const int batch_n_x,
    const int batch_n_y,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    const float max
) {
    auto warp_thread_index = threadIdx.x % WARP_WIDTH;
    auto global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    auto item_index = global_thread_index / WARP_WIDTH;

    if (batch_n_x <= item_index) {
        return;
    }

    const float norm_x_i = norms_x[item_index];
    float* row_p = all_distances_buffer + item_index * batch_n_y;

    for (uint32_t c = warp_thread_index; c < batch_n_y; c += WARP_WIDTH) {
        row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[c];
    }
}

template <typename T>
__global__ void health_check_buffer_kernel(const T* buffer, const std::size_t size) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    if (buffer[i] == 0.321497891f) {
        printf("huh\n");
    }
}

template <typename T>
constexpr T divide_round_up(const T a, const T b) {
    return (a + b - 1) / b;
}

template <typename T>
void health_check_buffer(const T* buffer, const std::size_t size) {
    const auto N_THREADS = 1024;
    const auto n_blocks = divide_round_up<int32_t>(size, N_THREADS);

    // printf("Pre Health Check\n");
    health_check_buffer_kernel<<<n_blocks, N_THREADS>>>(buffer, size);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // printf("Post Health Check\n");
}

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
) {
    // printf("Checking norms_x\n");
    // health_check_buffer(norms_x, i + batch_n_x);
    //  printf("Checking norms_y\n");
    // health_check_buffer(norms_y, j + batch_n_y);
    //  printf("Checking all_distances_buffer\n");
    // health_check_buffer(all_distances_buffer, batch_n_x * batch_n_y);
    //  printf("Checking distances\n");
    // health_check_buffer(out_distances, i + batch_n_x);
    //  printf("Checking out_knn\n");
    // health_check_buffer(out_knn, i + batch_n_x);
    const auto max = std::numeric_limits<float>::max();

    // ===============
    // WARNING : It might be the case that batch_n_y % 32 == 0 is required, due to use of warp
    // primitives NOTE: Other solutions are also possible, but did not test yet
    // ===============

    const auto N_THREADS_PER_BLOCK = 1024;
    const auto N_THREADS_PER_ITEM = WARP_WIDTH;
    const auto ITEMS_PER_BLOCK = divide_round_up<int32_t>(N_THREADS_PER_BLOCK, N_THREADS_PER_ITEM);
    const auto n_blocks = divide_round_up<int32_t>(batch_n_x, ITEMS_PER_BLOCK);
    first_blas_kernel<<<n_blocks, N_THREADS_PER_BLOCK, 0, stream>>>(
        batch_n_x,
        batch_n_y,
        i,
        j,
        norms_x + i,
        norms_y + j,
        all_distances_buffer,
        out_distances + i,
        out_knn + i,
        max
    );
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void norms(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    const cudaStream_t stream
) {
    // printf("Checking norms_x\n");
    // health_check_buffer(norms_x, i + batch_n_x);
    //  printf("Checking norms_y\n");
    // health_check_buffer(norms_y, j + batch_n_y);
    //  printf("Checking all_distances_buffer\n");
    // health_check_buffer(all_distances_buffer, batch_n_x * batch_n_y);
    const auto max = std::numeric_limits<float>::max();

    // ===============
    // WARNING : It might be the case that batch_n_y % 32 == 0 is required, due to use of warp
    // primitives NOTE: Other solutions are also possible, but did not test yet
    // ===============

    const auto N_THREADS_PER_BLOCK = 256;
    const auto N_THREADS_PER_ITEM = WARP_WIDTH;
    const auto ITEMS_PER_BLOCK = divide_round_up<int32_t>(N_THREADS_PER_BLOCK, N_THREADS_PER_ITEM);
    const auto n_blocks = divide_round_up<int32_t>(batch_n_x, ITEMS_PER_BLOCK);
    norms_kernel<<<n_blocks, N_THREADS_PER_BLOCK, 0, stream>>>(
        batch_n_x, batch_n_y, norms_x + i, norms_y + j, all_distances_buffer, max
    );
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__device__ skmeans_distance_t<Quantization::f32> warp_calculate_distance_horizontal_dimensions(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector1,
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector2,
    const uint32_t num_dimensions,
    const ThreadContext& thread_context
) {
    // Assumes one warp collaborating
    // All threads in warp participate
    // Only first thread returns relevant distance
    skmeans_distance_t<Quantization::f32> distance = 0.0;
    for (size_t dimension_idx = thread_context.lane_id; dimension_idx < num_dimensions;
         dimension_idx += WARP_WIDTH) {
        skmeans_distance_t<Quantization::f32> to_multiply =
            vector1[dimension_idx] - vector2[dimension_idx];
        distance += to_multiply * to_multiply;
    }

    distance = warp_reduce_sum(distance);
    return distance;
};

__global__ void calculate_distance_to_current_centroids_kernel(
    const uint32_t n_x,
    const uint32_t d,
    const data_t* SKM_RESTRICT x,
    const data_t* SKM_RESTRICT y,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances
) {
    const auto thread_context = ThreadContext();

    size_t vector_index = thread_context.warp_id;

    if (n_x <= vector_index) {
        return;
    }

    auto query_p = y + (out_knn[vector_index] * d);
    auto data_p = x + (vector_index * d);
    auto distance =
        warp_calculate_distance_horizontal_dimensions(query_p, data_p, d, thread_context);

    if (thread_context.is_first_lane()) {
        out_distances[vector_index] = distance;
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
) {
    // printf("Checking x\n");
    // health_check_buffer(x, n_x * d);
    // printf("Checking y\n");
    // health_check_buffer(y, n_y * d);
    // printf("Checking out_knn\n");
    // health_check_buffer(out_knn, n_x);
    // printf("Checking out_distances\n");
    // health_check_buffer(out_distances, n_x);
    // printf("Finished health checks\n");
    const auto N_THREADS_PER_BLOCK = 256;
    const auto N_THREADS_PER_ITEM = WARP_WIDTH;
    const auto ITEMS_PER_BLOCK = divide_round_up<uint32_t>(N_THREADS_PER_BLOCK, N_THREADS_PER_ITEM);
    const auto n_blocks = divide_round_up<uint32_t>(n_x, ITEMS_PER_BLOCK);
    calculate_distance_to_current_centroids_kernel<<<n_blocks, N_THREADS_PER_BLOCK, 0, stream>>>(
        n_x, d, x, y, out_knn, out_distances
    );
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
template <uint32_t NUM_DIMENSIONS>
__device__ skmeans_distance_t<Quantization::f32>
calculate_distance_with_fixed_horizontal_dimensions(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector1,
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector2,
    const ThreadContext& thread_context
) {
    // Assumes one warp collaborating
    // All threads in warp participate
    // All thread returns relevant distance, (but should probably be only one of them)
    skmeans_distance_t<Quantization::f32> distance = 0.0;
#pragma unroll
    for (size_t dimension_idx = thread_context.lane_id; dimension_idx < NUM_DIMENSIONS;
         dimension_idx += WARP_WIDTH) {
        skmeans_distance_t<Quantization::f32> to_multiply =
            vector1[dimension_idx] - vector2[dimension_idx];
        distance += to_multiply * to_multiply;
    }
    distance = warp_reduce_sum(distance);
    distance = warp_broadcast(distance);
    return distance;
};

__device__ skmeans_distance_t<Quantization::f32> calculate_distance_with_horizontal_dimensions(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector1,
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT vector2,
    const uint32_t num_dimensions,
    const ThreadContext& thread_context
) {
    // Assumes one warp collaborating
    // All threads in warp participate
    // All thread returns relevant distance, (but should probably be only one of them)

    skmeans_distance_t<Quantization::f32> distance = 0.0;

    for (size_t dimension_idx = thread_context.lane_id; dimension_idx < num_dimensions;
         dimension_idx += WARP_WIDTH) {
        skmeans_distance_t<Quantization::f32> to_multiply =
            vector1[dimension_idx] - vector2[dimension_idx];
        distance += to_multiply * to_multiply;
    }
    distance = warp_reduce_sum(distance);
    distance = warp_broadcast(distance);
    return distance;
};

__device__ void initialize_pruning_positions_array(
    size_t& n_vectors_not_pruned,
    const skmeans_distance_t<Quantization::f32> pruning_threshold,
    uint32_t* pruning_positions,
    skmeans_distance_t<Quantization::f32>* pruning_distances,
    const size_t n_vectors,
    const ThreadContext& thread_context
) {
    n_vectors_not_pruned = 0;
    for (size_t vector_idx = thread_context.lane_id; vector_idx < n_vectors;
         vector_idx += WARP_WIDTH) {
        auto distance = pruning_distances[vector_idx];
        bool distance_is_under_threshold = distance < pruning_threshold;

        auto old_n_vectors_not_pruned = n_vectors_not_pruned;
        auto offset = warp_exclusive_prefix_sum_binary_and_total(
            n_vectors_not_pruned, distance_is_under_threshold, __activemask()
        );
        if (distance_is_under_threshold) {
            pruning_positions[old_n_vectors_not_pruned + offset] = vector_idx;
        }
    }
    n_vectors_not_pruned = warp_broadcast(n_vectors_not_pruned);
}

__device__ void update_pruning_positions_array(
    size_t& n_vectors_not_pruned,
    const skmeans_distance_t<Quantization::f32> pruning_threshold,
    uint32_t* pruning_positions,
    skmeans_distance_t<Quantization::f32>* pruning_distances,
    const ThreadContext& thread_context
) {
    const auto previous_n_vectors_not_pruned = n_vectors_not_pruned;

    n_vectors_not_pruned = 0;
    for (size_t vector_idx = thread_context.lane_id; vector_idx < previous_n_vectors_not_pruned; vector_idx += WARP_WIDTH) {
        auto position = pruning_positions[vector_idx];
        auto distance = pruning_distances[position];

        bool distance_is_under_threshold = distance < pruning_threshold;

        auto old_n_vectors_not_pruned = n_vectors_not_pruned;
        auto offset = warp_exclusive_prefix_sum_binary_and_total(
            n_vectors_not_pruned, distance_is_under_threshold, __activemask()
        );
        if (distance_is_under_threshold) {
            pruning_positions[old_n_vectors_not_pruned + offset] = position;
        }
    }
    n_vectors_not_pruned = warp_broadcast(n_vectors_not_pruned);
}

__device__ void select_closest_vector(
    KNNCandidate<Quantization::f32>& best_candidate,
    const size_t n_vectors_not_pruned,
    const uint32_t* SKM_RESTRICT vector_indices,
    const uint32_t* SKM_RESTRICT pruning_positions,
    const skmeans_distance_t<Quantization::f32>* SKM_RESTRICT pruning_distances,
		const ThreadContext& thread_context
) {
    // TODO For all kernels, size_t is not necessary here

    // for (size_t position_idx = 0; position_idx < n_vectors_not_pruned; ++position_idx) {
    //     size_t index = pruning_positions[position_idx];
    //     auto current_distance = pruning_distances[index];
    //     if (current_distance < best_candidate.distance) {
    //         best_candidate.distance = current_distance;
    //         best_candidate.index = vector_indices[index];
    //     }
    // }

		// Not sure there are enough n_vectors_not_pruned to actually make collaboratively doing this worth it
    for (size_t position_idx = thread_context.lane_id; position_idx < n_vectors_not_pruned; position_idx += WARP_WIDTH) {
        size_t index = pruning_positions[position_idx];
        auto distance = pruning_distances[index];
        if (distance < best_candidate.distance) {
            best_candidate.distance = distance;
            best_candidate.index = vector_indices[index];
        }
    }

    auto result = index_min_warp_reduction(best_candidate.index, best_candidate.distance);
		best_candidate.index = warp_broadcast(result.index);
		best_candidate.distance = warp_broadcast(result.value);
}

template <uint32_t WARPS_PER_BLOCK>
__device__ void prune(
    const skmeans_value_t<Quantization::f32>* SKM_RESTRICT query,
    // const size_t y_batch,
    const ConstantPruneDataView constant_prune_data,
    const ClusterPruneDataView cluster_prune_data,
    skmeans_distance_t<Quantization::f32>* pruning_distances,
    KNNCandidate<Quantization::f32>& best_candidate,
    uint32_t current_dimension_idx,
    size_t& initial_not_pruned_accum,
    const ThreadContext& thread_context
) {
    // TODO Make shared
    // alignas(64)
    __shared__ uint32_t block_pruning_positions[PDX_VECTOR_SIZE * WARPS_PER_BLOCK];
    auto pruning_positions =
        block_pruning_positions + PDX_VECTOR_SIZE * thread_context.block_warp_id;

    const auto prev_best_candidate_distance = best_candidate.distance;

    auto pruning_threshold =
        prev_best_candidate_distance * constant_prune_data.ratios[current_dimension_idx];
    size_t n_vectors_not_pruned = 0;

    initialize_pruning_positions_array(
        n_vectors_not_pruned,
        pruning_threshold,
        pruning_positions,
        pruning_distances,
        cluster_prune_data.n_vectors,
        thread_context
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
                pruning_distances[v_idx] +=
                    calculate_distance_with_fixed_horizontal_dimensions<H_DIM_SIZE>(
                        query + offset_query, cluster_prune_data.data + data_pos, thread_context
                    );
            }

            current_dimension_idx += H_DIM_SIZE;
            pruning_threshold =
                prev_best_candidate_distance * constant_prune_data.ratios[current_dimension_idx];
            update_pruning_positions_array(
                n_vectors_not_pruned, pruning_threshold, pruning_positions, pruning_distances, thread_context
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
            pruning_distances[v_idx] += calculate_distance_with_horizontal_dimensions(
                query + offset_query, data_pos, dimensions_left, thread_context
            );
        }
        current_dimension_idx = constant_prune_data.num_dimensions;
        pruning_threshold =
            prev_best_candidate_distance * constant_prune_data.ratios[current_dimension_idx];
        update_pruning_positions_array(
            n_vectors_not_pruned, pruning_threshold, pruning_positions, pruning_distances, thread_context
        );
    }

    if (n_vectors_not_pruned) {
        select_closest_vector(
            best_candidate,
            n_vectors_not_pruned,
            cluster_prune_data.vector_indices,
            pruning_positions,
            pruning_distances,
						thread_context
        );
    }
}

template <uint32_t WARPS_PER_BLOCK>
__global__ void search_closest_centroid_with_pruning_kernel(
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const uint32_t partial_d,
    const data_t* SKM_RESTRICT x,
    const ConstantPruneDataView constant_prune_data,
    const ClusterPruneDataView cluster_prune_data,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    size_t* SKM_RESTRICT out_not_pruned_counts,
    float* SKM_RESTRICT all_distances_buf
) {
    auto thread_context = ThreadContext();
    auto r = thread_context.warp_id;

    if (batch_n_x <= r) {
        return;
    }

    auto data_p = x + (r * d);

    knn_candidate_t assigned_centroid{out_knn[r], out_distances[r]};

    // PDXearch per vector
    auto partial_distances_p = all_distances_buf + r * batch_n_y;
    size_t local_not_pruned = 0;

    prune<WARPS_PER_BLOCK>(
        data_p,
        constant_prune_data,
        cluster_prune_data,
        partial_distances_p,
        assigned_centroid,
        partial_d,
        local_not_pruned,
        thread_context
    );

    if (thread_context.is_first_lane()) {
        out_not_pruned_counts[r] += local_not_pruned;
        out_knn[r] = assigned_centroid.index;
        out_distances[r] = assigned_centroid.distance;
    }
}

void GPUSearchPDX(
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const uint32_t partial_d,
    const data_t* SKM_RESTRICT x,
    const ConstantPruneDataView constant_prune_data,
    const ClusterPruneDataView cluster_prune_data,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    size_t* SKM_RESTRICT out_not_pruned_counts,
    float* SKM_RESTRICT all_distances_buf,
    const cudaStream_t stream
) {
    const auto N_THREADS_PER_BLOCK = 256;
    const auto WARPS_PER_BLOCK = divide_round_up<int32_t>(N_THREADS_PER_BLOCK, WARP_WIDTH);
    const auto N_THREADS_PER_ITEM = WARP_WIDTH;
    const auto ITEMS_PER_BLOCK = divide_round_up<int32_t>(N_THREADS_PER_BLOCK, N_THREADS_PER_ITEM);
    const auto n_blocks = divide_round_up<int32_t>(batch_n_x, ITEMS_PER_BLOCK);

    search_closest_centroid_with_pruning_kernel<WARPS_PER_BLOCK>
        <<<n_blocks, N_THREADS_PER_BLOCK, 0, stream>>>(
            batch_n_x,
            batch_n_y,
            d,
            partial_d,
            x,
            constant_prune_data,
            cluster_prune_data,
            out_knn,
            out_distances,
            out_not_pruned_counts,
            all_distances_buf
        );
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

} // namespace kernels
} // namespace skmeans
