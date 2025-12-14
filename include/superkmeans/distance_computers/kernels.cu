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
    auto r = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_n_x <= r) {
        return;
    }

    const auto i_idx = i + r;
    const float norm_x_i = norms_x[i_idx];
    float* row_p = all_distances_buffer + r * batch_n_y;

    for (std::size_t c = 0; c < batch_n_y; ++c) {
        row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
    }
		uint32_t knn_idx;
    auto batch_top_1 = find_min(row_p, batch_n_y, max, &knn_idx);
    if (batch_top_1 < out_distances[i_idx]) {
        out_distances[i_idx] = max_of_either<float>(0.0f, batch_top_1);
        out_knn[i_idx] = j + knn_idx;
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

    const auto N_THREADS = 1024;
    const auto n_blocks = divide_round_up<int32_t>(batch_n_x, N_THREADS);
    first_blas_kernel<<<n_blocks, N_THREADS, 0, stream>>>(
        batch_n_x, batch_n_y, i, j, norms_x, norms_y, all_distances_buffer, out_distances, out_knn, max
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
