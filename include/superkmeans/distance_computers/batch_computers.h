#pragma once

#include <cstdint>
#include <cstdio>
#include <omp.h>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"

#include "superkmeans/distance_computers/kernels.cuh"
#include "superkmeans/gpu/gpu.cuh"

#include <Eigen/Eigen/Dense>

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Eigen already declares sgemm_, so we don't need to redeclare it
// Use Eigen's BLAS declarations from Eigen/src/Core/util/BlasUtil.h

namespace skmeans {
using distance_t = skmeans_distance_t<Quantization::f32>;
using data_t = skmeans_value_t<Quantization::f32>;
using norms_t = skmeans_value_t<Quantization::f32>;
using knn_candidate_t = KNNCandidate<Quantization::f32>;
using layout_t = PDXLayout<Quantization::f32, DistanceFunction::l2>;
using MatrixR = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixC = Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;


template <DistanceFunction alpha, Quantization q>
class BatchComputer {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::u8> {};

template <>
class BatchComputer<DistanceFunction::l2, Quantization::f32> {

  private:
/**
 * @brief Performs BLAS matrix multiplication: distances = x * y^T
 * Note that Eigen internally uses BLAS for matrix multiplication, so we can use it directly.
 * This is convenient as it will work even if a BLAS library is not available.
 *
 * Computes the dot product matrix between query vectors (x) and reference vectors (y).
 * Can optionally use only the first partial_d dimensions for partial distance computation.
 *
 * @param batch_x_p Pointer to query vectors batch (batch_n_x × d)
 * @param batch_y_p Pointer to reference vectors batch (batch_n_y × d)
 * @param batch_n_x Number of query vectors in batch
 * @param batch_n_y Number of reference vectors in batch
 * @param d Full dimensionality
 * @param partial_d Number of dimensions to use (if 0 or d, uses all dimensions)
 * @param all_distances_buf Output buffer for distance matrix (batch_n_x × batch_n_y)
 */
static void
BlasMatrixMultiplication(
    const data_t* SKM_RESTRICT batch_x_p,
    const data_t* SKM_RESTRICT batch_y_p,
    const size_t batch_n_x,
    const size_t batch_n_y,
    const size_t d,
    const size_t partial_d,
    float* SKM_RESTRICT all_distances_buf
) {
    // Direct BLAS sgemm implementation
    // Compute: distances = x * y^T (all row-major)
    // where x is batch_n_x × d, y is batch_n_y × d, distances is batch_n_x × batch_n_y
    //
    // For row-major matrices with column-major BLAS (Fortran interface):
    // To compute C = A * B^T, we call: sgemm('T', 'N', n, m, k, alpha, B, ldb, A, lda, beta, C,
    // ldc) This is the standard row-major to column-major translation

    const char trans_a = 'T'; // Transpose flag for first operand
    const char trans_b = 'N'; // No transpose for second operand

    int m = static_cast<int>(batch_n_y); // Rows of result (swapped for row-major)
    int n = static_cast<int>(batch_n_x); // Cols of result (swapped for row-major)
    int k = static_cast<int>(partial_d > 0 && partial_d < d ? partial_d : d); // Inner dimension

    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = static_cast<int>(d);         // Leading dimension of y (row stride in row-major)
    int ldb = static_cast<int>(d);         // Leading dimension of x (row stride in row-major)
    int ldc = static_cast<int>(batch_n_y); // Leading dimension of distances

    sgemm_(
        &trans_a,
        &trans_b,
        &m,
        &n,
        &k,
        &alpha,
        batch_y_p,
        &lda, // y is first operand (transposed)
        batch_x_p,
        &ldb, // x is second operand (not transposed)
        &beta,
        all_distances_buf,
        &ldc
    );

    // Old Eigen implementation (commented out):
    // Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
    // Eigen::Map<const MatrixR> x_matrix(batch_x_p, batch_n_x, d);
    // Eigen::Map<const MatrixR> y_matrix(batch_y_p, batch_n_y, d);
    //
    // if (partial_d > 0 && partial_d < d) {
    //     // Partial multiplication: use only first partial_d dimensions
    //     distances_matrix.noalias() =
    //         x_matrix.leftCols(partial_d) * y_matrix.leftCols(partial_d).transpose();
    // } else {
    //     // Full multiplication: use all dimensions
    //     distances_matrix.noalias() = x_matrix * y_matrix.transpose();
    // }
}

public:
/**
 * @brief Finds the nearest neighbor for each query vector using batched BLAS.
 *
 * Computes L2 distances between all query vectors (X) and reference vectors (Y)
 * using the identity: ||x-y||² = ||x||² + ||y||² - 2*x·y
 * The dot products are computed efficiently via matrix multiplication.
 *
 * @param x Query vectors in row-major layout (n_x × d)
 * @param y Reference vectors in row-major layout (n_y × d)
 * @param n_x Number of query vectors
 * @param n_y Number of reference vectors
 * @param d Dimensionality
 * @param norms_x Pre-computed squared L2 norms of query vectors
 * @param norms_y Pre-computed squared L2 norms of reference vectors
 * @param out_knn Output: index of nearest neighbor for each query
 * @param out_distances Output: distance to nearest neighbor for each query
 * @param all_distances_buf Scratch buffer for batch distance computation (size: X_BATCH_SIZE ×
 * Y_BATCH_SIZE)
 */
static void FindNearestNeighbor(
    const data_t* SKM_RESTRICT x,
    const data_t* SKM_RESTRICT y,
    const size_t n_x,
    const size_t n_y,
    const size_t d,
    const norms_t* SKM_RESTRICT norms_x,
    const norms_t* SKM_RESTRICT norms_y,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    float* SKM_RESTRICT all_distances_buf
) {
    SKM_PROFILE_SCOPE("1st_blas");

    std::fill_n(out_distances, n_x, std::numeric_limits<distance_t>::max());

    auto stream = gpu::ManagedCudaStream();
		auto y_dev = gpu::DeviceBuffer<data_t>(gpu::compute_buffer_size<data_t>(n_y, d), stream.get());
		y_dev.copy_to_device(y);
    auto multiplier = gpu::BatchedMatrixMultiplier(stream.get());

		constexpr int32_t N_BATCH_STREAMS = 4;
		auto batch_streams = std::vector<gpu::ManagedCudaStream>(N_BATCH_STREAMS);
		auto batch_multipliers = std::vector<gpu::BatchedMatrixMultiplier>();
		auto batch_x_buffers_dev = std::vector<gpu::DeviceBuffer<data_t>>();
		auto batch_all_distances_buffers_dev = std::vector<gpu::DeviceBuffer<float>>();

		batch_multipliers.reserve(N_BATCH_STREAMS);
		batch_all_distances_buffers_dev.reserve(N_BATCH_STREAMS);

		for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
			batch_multipliers.emplace_back(batch_streams[i].get());
			batch_x_buffers_dev.emplace_back(gpu::compute_buffer_size<float>(X_BATCH_SIZE, d),batch_streams[i].get());
			batch_all_distances_buffers_dev.emplace_back(gpu::compute_buffer_size<float>(X_BATCH_SIZE, Y_BATCH_SIZE),batch_streams[i].get());
		}

		auto norms_x_dev = gpu::DeviceBuffer<norms_t>(gpu::compute_buffer_size<norms_t>(n_x),stream.get());
		auto norms_y_dev = gpu::DeviceBuffer<norms_t>(gpu::compute_buffer_size<norms_t>(n_y),stream.get());
		auto out_distances_dev = gpu::DeviceBuffer<distance_t>(gpu::compute_buffer_size<distance_t>(n_x),stream.get());
		auto out_knn_dev = gpu::DeviceBuffer<uint32_t>(gpu::compute_buffer_size<uint32_t>(n_x),stream.get());
		norms_x_dev.copy_to_device(norms_x);
		norms_y_dev.copy_to_device(norms_y);
		//all_distances_buf_dev.copy_to_device(all_distances_buf);
		out_distances_dev.copy_to_device(out_distances);
		out_knn_dev.copy_to_device(out_knn);
		stream.synchronize();

		size_t iteration_count = 0;
    for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
        auto batch_n_x = X_BATCH_SIZE;
        auto batch_x_p = x + (i * d);
        if (i + X_BATCH_SIZE > n_x) {
            batch_n_x = n_x - i;
        }

				auto batch_stream_index = iteration_count % N_BATCH_STREAMS;
				++iteration_count;

				batch_x_buffers_dev[batch_stream_index].copy_to_device(batch_x_p, gpu::compute_buffer_size<data_t>(batch_n_x, d));
        for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
            auto batch_n_y = Y_BATCH_SIZE;
            auto batch_y_p = y_dev.get() + (j * d);
            if (j + Y_BATCH_SIZE > n_y) {
                batch_n_y = n_y - j;
            }
						batch_multipliers[batch_stream_index].multiply(batch_x_buffers_dev[batch_stream_index].get(), batch_y_p, batch_n_x, batch_n_y, d, 0, 
								batch_all_distances_buffers_dev[batch_stream_index].get());

						kernels::first_blas(
								batch_n_x,
								batch_n_y,
								i, 
								j,
								norms_x_dev.get(),
								norms_y_dev.get(),
								batch_all_distances_buffers_dev[batch_stream_index].get(),
								out_distances_dev.get(),
								out_knn_dev.get(),
								batch_streams[batch_stream_index].get()
								);
        }
    }
		stream.synchronize();
		for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
			batch_streams[i].synchronize();
		}
		out_distances_dev.copy_to_host(out_distances);
		out_knn_dev.copy_to_host(out_knn);
		stream.synchronize();
}

/**
 * @brief Finds the k nearest neighbors for each query vector using batched BLAS.
 *
 * Similar to FindNearestNeighbor but maintains top-k candidates per query.
 * Results are merged across Y batches using partial sort.
 *
 * @param x Query vectors in row-major layout (n_x × d)
 * @param y Reference vectors in row-major layout (n_y × d)
 * @param n_x Number of query vectors
 * @param n_y Number of reference vectors
 * @param d Dimensionality
 * @param norms_x Pre-computed squared L2 norms of query vectors
 * @param norms_y Pre-computed squared L2 norms of reference vectors
 * @param k Number of nearest neighbors to find
 * @param out_knn Output: indices of k nearest neighbors for each query (size: n_x × k)
 * @param out_distances Output: distances to k nearest neighbors (size: n_x × k)
 * @param all_distances_buf Scratch buffer for batch distance computation
 */
static void FindKNearestNeighbors(
    const data_t* SKM_RESTRICT x,
    const data_t* SKM_RESTRICT y,
    const size_t n_x,
    const size_t n_y,
    const size_t d,
    const norms_t* SKM_RESTRICT norms_x,
    const norms_t* SKM_RESTRICT norms_y,
    const size_t k,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    float* SKM_RESTRICT all_distances_buf
) {
    // Initialize output with infinity
    std::fill_n(out_distances, n_x * k, std::numeric_limits<distance_t>::max());
    std::fill_n(out_knn, n_x * k, static_cast<uint32_t>(-1));

    // Pre-allocate per-thread candidate buffers to avoid heap allocation in the hot loop
    const size_t max_candidates = k + Y_BATCH_SIZE;
    const uint32_t num_threads = g_n_threads;
    std::vector<std::vector<std::pair<float, uint32_t>>> thread_candidates(num_threads);
    for (auto& tc : thread_candidates) {
        tc.reserve(max_candidates);
    }

    for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
        auto batch_n_x = X_BATCH_SIZE;
        auto batch_x_p = x + (i * d);
        if (i + X_BATCH_SIZE > n_x) {
            batch_n_x = n_x - i;
        }
        for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
            auto batch_n_y = Y_BATCH_SIZE;
            auto batch_y_p = y + (j * d);
            if (j + Y_BATCH_SIZE > n_y) {
                batch_n_y = n_y - j;
            }

            BlasMatrixMultiplication(
                batch_x_p, batch_y_p, batch_n_x, batch_n_y, d, 0, all_distances_buf
            );
            Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);

#pragma omp parallel for num_threads(g_n_threads)
            for (size_t r = 0; r < batch_n_x; ++r) {
                const auto i_idx = i + r;
                const float norm_x_i = norms_x[i_idx];
                float* row_p = distances_matrix.data() + r * batch_n_y;

                // Compute L2 distances for current batch
#pragma clang loop vectorize(enable)
                for (size_t c = 0; c < batch_n_y; ++c) {
                    row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                }

                // Merge: Combine previous top-k with current Y batch candidates
                // Use pre-allocated per-thread buffer instead of allocating each iteration
                auto& candidates = thread_candidates[omp_get_thread_num()];
                candidates.clear();

                // Add previous top-k (skip if distance is infinity, meaning not filled yet)
                for (size_t ki = 0; ki < k; ++ki) {
                    if (out_distances[i_idx * k + ki] < std::numeric_limits<distance_t>::max()) {
                        candidates.push_back(
                            {out_distances[i_idx * k + ki], out_knn[i_idx * k + ki]}
                        );
                    }
                }

                // Add current batch candidates
                for (size_t c = 0; c < batch_n_y; ++c) {
                    candidates.push_back({row_p[c], static_cast<uint32_t>(j + c)});
                }

                // Partial sort to get new top-k
                size_t actual_k = std::min(k, candidates.size());
                std::partial_sort(
                    candidates.begin(), candidates.begin() + actual_k, candidates.end()
                );

                // Update output with new top-k
                for (size_t ki = 0; ki < actual_k; ++ki) {
                    out_distances[i_idx * k + ki] = std::max(0.0f, candidates[ki].first);
                    out_knn[i_idx * k + ki] = candidates[ki].second;
                }

                // Fill remaining slots if needed
                for (size_t ki = actual_k; ki < k; ++ki) {
                    out_distances[i_idx * k + ki] = std::numeric_limits<distance_t>::max();
                    out_knn[i_idx * k + ki] = static_cast<uint32_t>(-1);
                }
            }
        }
    }
}



/**
 * @brief Finds nearest neighbors using partial BLAS computation with PDX pruning.
 *
 * Hybrid approach that computes partial distances (first partial_d dimensions)
 * via BLAS, then uses ADSampling pruning to skip full distance computation
 * for unlikely candidates. Significantly faster than full BLAS when pruning
 * is effective. To prune even better, we get a threshold from the previously assigned centroid.
 *
 * @param x Query vectors in row-major layout (n_x × d)
 * @param y Reference vectors in row-major layout (n_y × d)
 * @param n_x Number of query vectors
 * @param n_y Number of reference vectors (centroids)
 * @param d Full dimensionality
 * @param norms_x Pre-computed partial squared L2 norms of queries (first partial_d dims)
 * @param norms_y Pre-computed partial squared L2 norms of references (first partial_d dims)
 * @param out_knn Input/Output: current assignment indices (updated with better assignments)
 * @param out_distances Input/Output: current distances (updated with better distances)
 * @param all_distances_buf Scratch buffer for batch distance computation
 * @param pdx_centroids PDX layout containing centroids and searcher for pruned search
 * @param partial_d Number of dimensions used for initial BLAS computation
 * @param out_not_pruned_counts Optional output: count of non-pruned vectors per query (for
 * tuning)
 */

struct StreamBuffers {
		cudaStream_t stream;
		gpu::DeviceBuffer<data_t> batch_x_buffer_dev; 
		gpu::DeviceBuffer<norms_t> all_distances_buf_dev; 
    gpu::BatchedMatrixMultiplier multiplier; 

		StreamBuffers(
				const size_t x_batch_size,
				const size_t y_batch_size,
				const size_t d,
				const cudaStream_t stream
				) : 
					stream(stream),
					batch_x_buffer_dev(gpu::compute_buffer_size<data_t>(x_batch_size, d),stream),
					all_distances_buf_dev(gpu::compute_buffer_size<float>(x_batch_size, y_batch_size),stream),
					multiplier(stream) {}
};

static void FindNearestNeighborWithPruning(
    const data_t* SKM_RESTRICT x,
    const data_t* SKM_RESTRICT y,
    const size_t n_x,
    const size_t n_y,
    const size_t d,
    const norms_t* SKM_RESTRICT norms_x,
    const norms_t* SKM_RESTRICT norms_y,
    uint32_t* SKM_RESTRICT out_knn,
    distance_t* SKM_RESTRICT out_distances,
    float* SKM_RESTRICT all_distances_buf,
    const layout_t& pdx_centroids,
    uint32_t partial_d,
    size_t* out_not_pruned_counts = nullptr
) {
    SKM_PROFILE_SCOPE("search");
    auto stream = gpu::ManagedCudaStream();
		auto y_dev = gpu::DeviceBuffer<data_t>(gpu::compute_buffer_size<data_t>(n_y, d), stream.get());
		y_dev.copy_to_device(y);

		auto norms_x_dev = gpu::DeviceBuffer<norms_t>(gpu::compute_buffer_size<norms_t>(n_x),stream.get());
		auto norms_y_dev = gpu::DeviceBuffer<norms_t>(gpu::compute_buffer_size<norms_t>(n_y),stream.get());
		norms_x_dev.copy_to_device(norms_x);
		norms_y_dev.copy_to_device(norms_y);

		auto out_knn_dev = gpu::DeviceBuffer<uint32_t>(gpu::compute_buffer_size<uint32_t>(n_x),stream.get()); 
		out_knn_dev.copy_to_device(out_knn);
		auto out_distances_dev = gpu::DeviceBuffer<distance_t>(gpu::compute_buffer_size<distance_t>(n_x),stream.get()); 
		auto out_not_pruned_counts_dev = gpu::DeviceBuffer<size_t>(gpu::compute_buffer_size<size_t>(n_x),stream.get()); 
		out_not_pruned_counts_dev.copy_to_device(out_not_pruned_counts);

		auto constant_prune_data = kernels::ConstantPruneData(pdx_centroids, stream.get());

		const size_t n_y_clusters = (n_y + Y_BATCH_SIZE - 1) / Y_BATCH_SIZE;
		auto cluster_data = std::vector<kernels::ClusterPruneData>();
		cluster_data.reserve(n_y_clusters);

		for (size_t i = 0; i < n_y_clusters; ++i) {
			auto y_batch_cluster = pdx_centroids.searcher->pdx_data.clusters[i];
			cluster_data.emplace_back(y_batch_cluster, constant_prune_data.as_view(), n_y, stream.get());
		}

		constexpr int32_t N_BATCH_STREAMS = 4;
		auto batch_streams = std::vector<gpu::ManagedCudaStream>(N_BATCH_STREAMS);
		auto batch_stream_buffers = std::vector<StreamBuffers>();
		batch_stream_buffers.reserve(N_BATCH_STREAMS);

		for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
			batch_stream_buffers.emplace_back(
					X_BATCH_SIZE,
					Y_BATCH_SIZE,
					d,
					batch_streams[i].get()
			);
		}

		stream.synchronize();

		size_t iteration_count = 0;
    for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
        auto batch_n_x = X_BATCH_SIZE;
        auto batch_x_p = x + (i * d);

        if (i + X_BATCH_SIZE > n_x) {
            batch_n_x = n_x - i;
        }

				auto current_stream = iteration_count % N_BATCH_STREAMS;
				iteration_count += 1;

				batch_stream_buffers[current_stream].batch_x_buffer_dev.copy_to_device(batch_x_p, gpu::compute_buffer_size<data_t>(batch_n_x, d));

				kernels::GPUCalculateDistanceToCurrentCentroids(
					batch_n_x,
					n_y,
					d,
					batch_stream_buffers[current_stream].batch_x_buffer_dev.get(),
					y_dev.get(), // const
					out_knn_dev.get() + i, // const
					out_distances_dev.get() + i, // mutated
					batch_stream_buffers[current_stream].stream);

        MatrixR materialize_x_left_cols;
				size_t current_y_batch = 0;
        for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
            auto batch_n_y = Y_BATCH_SIZE;
            //auto batch_y_p = y + (j * d);
            auto batch_y_p = y_dev.get() + (j * d);
            if (j + Y_BATCH_SIZE > n_y) {
                batch_n_y = n_y - j;
            }

						batch_stream_buffers[current_stream].multiplier.multiply(
							batch_stream_buffers[current_stream].batch_x_buffer_dev.get(), // const
							batch_y_p, // const
							batch_n_x, 
							batch_n_y, 
							d, 
							partial_d, 
							batch_stream_buffers[current_stream].all_distances_buf_dev.get() // mutated
						);
						kernels::norms(
							batch_n_x,
							batch_n_y,
							i,
							j,
							norms_x_dev.get(), // const
							norms_y_dev.get(), // const
							batch_stream_buffers[current_stream].all_distances_buf_dev.get(), // mutated
							batch_stream_buffers[current_stream].stream
						);

						kernels::GPUSearchPDX(
							batch_n_x,
							batch_n_y,
							d,
							partial_d,
							batch_stream_buffers[current_stream].batch_x_buffer_dev.get(), // const
							constant_prune_data.as_view(), // const
							cluster_data[current_y_batch].as_view(), // const
							out_knn_dev.get() + i, // mutated
							out_distances_dev.get() + i, // mutated
							out_not_pruned_counts_dev.get() + i, // mutated
							batch_stream_buffers[current_stream].all_distances_buf_dev.get(), // mutated
							batch_stream_buffers[current_stream].stream); 

						current_y_batch += 1;
        }
    }
		for (int32_t i{0}; i < N_BATCH_STREAMS; ++i) {
			batch_streams[i].synchronize();
		}
		out_knn_dev.copy_to_host(out_knn);
		out_distances_dev.copy_to_host(out_distances);
		out_not_pruned_counts_dev.copy_to_host(out_not_pruned_counts);
		stream.synchronize();
}
};

} // namespace skmeans
