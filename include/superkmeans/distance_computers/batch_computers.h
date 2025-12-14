#pragma once

#include <cstdint>
#include <cstdio>
#include <omp.h>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/profiler.h"
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

namespace gpu {

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

class ManagedCudaStream {
  public:
    ManagedCudaStream() { CUDA_SAFE_CALL(cudaStreamCreate(&_stream)); }

    ~ManagedCudaStream() { CUDA_SAFE_CALL(cudaStreamDestroy(_stream)); }

    void synchronize() { CUDA_SAFE_CALL(cudaStreamSynchronize(_stream)); }

    cudaStream_t get() const { return _stream; }

  private:
    cudaStream_t _stream;
};

class ManagedCublasHandle {
  public:
    ManagedCublasHandle(cudaStream_t stream) {
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }
    ~ManagedCublasHandle() { cublasDestroy(handle); }

    cublasHandle_t handle;
};

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer(std::size_t size, cudaStream_t stream) : _size(size), _stream(stream) {
        CUDA_SAFE_CALL(cudaMalloc(&_dev_ptr, _size));
    }

    ~DeviceBuffer() {
        if (_dev_ptr) {
            CUDA_SAFE_CALL(cudaFree(_dev_ptr));
        };
    }

    void copy_to_device(const T* host_ptr) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(_dev_ptr),
            reinterpret_cast<const void*>(host_ptr),
            _size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_device(const T* host_ptr, const std::size_t size) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(_dev_ptr),
            reinterpret_cast<const void*>(host_ptr),
            size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_host(T* host_ptr) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(host_ptr),
            reinterpret_cast<const void*>(_dev_ptr),
            _size,
            cudaMemcpyDeviceToHost,
            _stream
        ));
    }

    void copy_to_host(T* host_ptr, const std::size_t size) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(host_ptr),
            reinterpret_cast<const void*>(_dev_ptr),
            size,
            cudaMemcpyDeviceToHost,
            _stream
        ));
    }

    T* get() const { return _dev_ptr; }

  private:
    T* _dev_ptr{nullptr};
    std::size_t _size;
    cudaStream_t _stream;
};

template <typename T>
std::size_t compute_buffer_size(const std::size_t x, const std::size_t y) {
		return x * y * sizeof(T);
}

class BatchedMatrixMultiplier {

  public:
    BatchedMatrixMultiplier(
        const std::size_t max_batch_n_x,
        const std::size_t max_batch_n_y,
        const std::size_t d,
        const cudaStream_t stream
    )
        : _cublas_handle(stream),
          _batch_x_dev_p(compute_buffer_size<data_t>(max_batch_n_x, d), stream),
          _batch_y_dev_p(compute_buffer_size<data_t>(max_batch_n_y, d), stream),
          _all_distances_dev_p(compute_buffer_size<float>(max_batch_n_y, max_batch_n_x), stream)
    {}

		void load_x_batch(
        const data_t* SKM_RESTRICT batch_x_p,
        const size_t batch_n_x,
        const size_t d) {
        const auto size_x = compute_buffer_size<data_t>(batch_n_x, d);
        _batch_x_dev_p.copy_to_device(batch_x_p, size_x);
		}

    void multiply(
        //const data_t* SKM_RESTRICT batch_x_p,
        const data_t* SKM_RESTRICT batch_y_p,
        const size_t batch_n_x,
        const size_t batch_n_y,
        const size_t d,
        float* SKM_RESTRICT all_distances_buf
    ) {
        const int m(static_cast<int>(batch_n_y)); // Rows of result (swapped for row-major)
        const int n(static_cast<int>(batch_n_x)); // Cols of result (swapped for row-major)
        const int k(static_cast<int>(d));         // Inner dimension

        const int lda(static_cast<int>(d)); // Leading dimension of y (row stride in row-major)
        const int ldb(static_cast<int>(d)); // Leading dimension of x (row stride in row-major)
        const int ldc(static_cast<int>(batch_n_y)); // Leading dimension of distances

        const auto size_y = compute_buffer_size<data_t>(batch_n_y, d);
        const auto size_out = compute_buffer_size<float>(batch_n_x, batch_n_y);

        _batch_y_dev_p.copy_to_device(batch_y_p, size_y);

        // TODO I think if you reverse A & B, you might not have to transpose at all
        cublasSgemm(
            _cublas_handle.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &ALPHA,
            _batch_y_dev_p.get(),
            ldb,
            _batch_x_dev_p.get(),
            lda,
            &BETA,
            _all_distances_dev_p.get(),
            ldc
        );

        _all_distances_dev_p.copy_to_host(all_distances_buf, size_out);
    }

    ~BatchedMatrixMultiplier() {}

  private:
    static constexpr float ALPHA = 1.0f;
    static constexpr float BETA = 0.0f;
    ManagedCublasHandle _cublas_handle;
    gpu::DeviceBuffer<data_t> _batch_x_dev_p;
    gpu::DeviceBuffer<data_t> _batch_y_dev_p;
    gpu::DeviceBuffer<float> _all_distances_dev_p;
};
}

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
    SKM_PROFILE_SCOPE("search");
    SKM_PROFILE_SCOPE("search/1st_blas");
    std::fill_n(out_distances, n_x, std::numeric_limits<distance_t>::max());

    auto stream = gpu::ManagedCudaStream();
    auto multiplier = gpu::BatchedMatrixMultiplier(X_BATCH_SIZE, Y_BATCH_SIZE, d, stream.get());

    for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
        auto batch_n_x = X_BATCH_SIZE;
        auto batch_x_p = x + (i * d);
        if (i + X_BATCH_SIZE > n_x) {
            batch_n_x = n_x - i;
        }
				multiplier.load_x_batch(batch_x_p, batch_n_x, d);
        for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
            auto batch_n_y = Y_BATCH_SIZE;
            auto batch_y_p = y + (j * d);
            if (j + Y_BATCH_SIZE > n_y) {
                batch_n_y = n_y - j;
            }
						multiplier.multiply(batch_y_p, batch_n_x, batch_n_y, d, all_distances_buf);
						stream.synchronize();
            Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
						// Idea: Make the rest of the loop into a kernel
						// Downside: this would add more data to be transferred to GPU, instead of from GPU (which is less contested)
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
            for (size_t r = 0; r < batch_n_x; ++r) {
                const auto i_idx = i + r;
                const float norm_x_i = norms_x[i_idx];
                float* row_p = distances_matrix.data() + r * batch_n_y;
#pragma clang loop vectorize(enable)
                for (size_t c = 0; c < batch_n_y; ++c) {
                    row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                }
                uint32_t knn_idx;
								// Calculate minimal value of row
                auto batch_top_1 = distances_matrix.row(r).minCoeff(&knn_idx);
                if (batch_top_1 < out_distances[i_idx]) {
                    out_distances[i_idx] = std::max(0.0f, batch_top_1);
                    out_knn[i_idx] = j + knn_idx;
                }
            }
        }
    }
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
    for (size_t i = 0; i < n_x; i += X_BATCH_SIZE) {
        auto batch_n_x = X_BATCH_SIZE;
        auto batch_x_p = x + (i * d);
        if (i + X_BATCH_SIZE > n_x) {
            batch_n_x = n_x - i;
        }
        MatrixR materialize_x_left_cols;
        for (size_t j = 0; j < n_y; j += Y_BATCH_SIZE) {
            auto batch_n_y = Y_BATCH_SIZE;
            auto batch_y_p = y + (j * d);
            if (j + Y_BATCH_SIZE > n_y) {
                batch_n_y = n_y - j;
            }
            {
                SKM_PROFILE_SCOPE("search/blas");
                BlasMatrixMultiplication(
                    batch_x_p, batch_y_p, batch_n_x, batch_n_y, d, partial_d, all_distances_buf
                );
            }
            Eigen::Map<MatrixR> distances_matrix(all_distances_buf, batch_n_x, batch_n_y);
            {
                SKM_PROFILE_SCOPE("search/norms");
#pragma omp parallel for num_threads(g_n_threads)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    const float norm_x_i = norms_x[i_idx];
                    float* row_p = distances_matrix.data() + r * batch_n_y;
#pragma clang loop vectorize(enable)
                    for (size_t c = 0; c < batch_n_y; ++c) {
                        row_p[c] = -2.0f * row_p[c] + norm_x_i + norms_y[j + c];
                    }
                }
            }
            {
                SKM_PROFILE_SCOPE("search/pdx");
#pragma omp parallel for num_threads(g_n_threads) schedule(dynamic, 8)
                for (size_t r = 0; r < batch_n_x; ++r) {
                    const auto i_idx = i + r;
                    auto data_p = x + (i_idx * d);
                    // Note that this will take the KNN from the previous batch loop
                    const auto prev_assignment = out_knn[i_idx];
                    distance_t dist_to_prev_centroid;
                    if (j == 0) { // After this out_distances always have the right distance
                        dist_to_prev_centroid =
                            DistanceComputer<DistanceFunction::l2, Quantization::f32>::Horizontal(
                                y + (prev_assignment * d), data_p, d
                            );
                    } else {
                        dist_to_prev_centroid = out_distances[i_idx];
                    }

                    // PDXearch per vector
                    knn_candidate_t assignment;
                    auto partial_distances_p = distances_matrix.data() + r * batch_n_y;
                    size_t local_not_pruned = 0;
                    assignment =
                        pdx_centroids.searcher->Top1PartialSearchWithThresholdAndPartialDistances(
                            data_p,
                            dist_to_prev_centroid,
                            prev_assignment,
                            partial_distances_p,
                            partial_d,
                            j / VECTOR_CHUNK_SIZE, // start cluster_id
                            (j + Y_BATCH_SIZE) /
                                VECTOR_CHUNK_SIZE, // end cluster_id; We use Y_BATCH_SIZE
                                                   // and not batch_n_y because otherwise we
                                                   // would not go up until incomplete
                                                   // clusters
                            out_not_pruned_counts != nullptr ? &local_not_pruned : nullptr
                        );
                    // Store not-pruned count for this X vector (accumulate across Y batches)
                    if (out_not_pruned_counts != nullptr) {
                        out_not_pruned_counts[i_idx] += local_not_pruned;
                    }
                    auto [assignment_idx, assignment_distance] = assignment;
                    out_knn[i_idx] = assignment_idx;
                    out_distances[i_idx] = assignment_distance;
                }
            }
        }
    }
}
};

} // namespace skmeans
