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

const int32_t WARP_WIDTH = 32;
const int32_t FULL_MASK = 0xffffffff;


template<typename T>
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

template<typename T>
__device__ constexpr T max_of_either(const T a, const T b) {
	if (a < b) {
		return b;
	}
	return a;
}

struct IndexMinWarpReductionResult {
	int32_t index;
	float value;
};

__device__ IndexMinWarpReductionResult index_min_warp_reduction(const int32_t index, const float value) {
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
    for (int32_t c = warp_thread_index; c < batch_n_y; c += WARP_WIDTH) {
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

template<typename T>
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

template<typename T>
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
	//printf("Checking norms_x\n");
	//health_check_buffer(norms_x, i + batch_n_x);
	// printf("Checking norms_y\n");
	//health_check_buffer(norms_y, j + batch_n_y);
	// printf("Checking all_distances_buffer\n");
	//health_check_buffer(all_distances_buffer, batch_n_x * batch_n_y);
	// printf("Checking distances\n");
	//health_check_buffer(out_distances, i + batch_n_x);
	// printf("Checking out_knn\n");
	//health_check_buffer(out_knn, i + batch_n_x);
		const auto max = std::numeric_limits<float>::max();

		// ===============
		// WARNING : It might be the case that batch_n_y % 32 == 0, due to use of warp primitives
		// NOTE: Other solutions are also possible, but did not test yet
		// ===============

    const auto N_THREADS_PER_BLOCK = 1024;
    const auto N_THREADS_PER_ITEM = WARP_WIDTH;
		const auto ITEMS_PER_BLOCK = divide_round_up<int32_t>(N_THREADS_PER_BLOCK, N_THREADS_PER_ITEM);
    const auto n_blocks = divide_round_up<int32_t>(batch_n_x, ITEMS_PER_BLOCK);
    first_blas_kernel<<<n_blocks, N_THREADS_PER_BLOCK, 0, stream>>>(
        batch_n_x, batch_n_y, i, j, norms_x + i, norms_y + j, all_distances_buffer, out_distances + i, out_knn + i, max
    );
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

/*
multiplier.multiply(batch_y_p, batch_n_x, batch_n_y, d, 0, all_distances_buf);
stream.synchronize();
Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
// Idea: Make the rest of the loop into a kernel
// Downside: this would add more data to be transferred to GPU, instead of from GPU (which is less
contested)
// So let's wait for now
// Outside vars
// - i
// - batch_n_x
// - batch_n_y
// - norms_x [needs to be batched into GPU]
// - norms_y [needs to be batched into GPU]
// - out_distances [needs to be batched out of GPU]
// - out_knn [needs to be batched out of GPU]
#pragma omp parallel for num_threads(g_n_threads)
*/
}
}
